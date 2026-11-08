using Bigtree.Algorithm.Internal;
using NumSharp;
using NumSharp.Backends.Iteration;
using System;

namespace Bigtree.Algorithm.KNN
{
    /// <summary>
    ///     k-nearest-neighbours classification with Euclidean distance and uniform weights: a sample gets the label
    ///     that most of its <see cref="K"/> nearest training samples carry. scikit-learn's
    ///     <c>KNeighborsClassifier(algorithm="brute")</c>, computed on NumSharp.
    /// </summary>
    /// <remarks>
    ///     <para><b>How it computes.</b> Brute force over whole arrays: the squared distances of a batch to every
    ///     training sample are <c>|x|^2 - 2 x . t + |t|^2</c>, one matrix product (scikit-learn computes them the same
    ///     way, here fused into one <c>np.evaluate</c> pass); <c>np.argpartition</c> picks the <see cref="K"/> smallest per
    ///     row without sorting the rest; the votes are one <c>np.bincount</c>. Queries run in blocks of at most 4,194,304
    ///     distances (32 MB), each in its own <see cref="NDScope"/>, so memory does not grow with the batch: 8,000 queries
    ///     against 8,000 training samples peaked at 3.0 GB computed at once.</para>
    ///     <para><b>Ties.</b> A vote tie goes to the class that comes first in <see cref="Classes"/> (ascending order),
    ///     like scikit-learn. Neighbours at exactly equal distance are an arbitrary pick between them, in scikit-learn
    ///     too.</para>
    ///     <para><b>Ownership.</b> The model keeps its own copy of the training data; dispose the model (or let the
    ///     finalizer do it) to return the buffers to NumSharp's pool.</para>
    /// </remarks>
    public class KnnModel : IDisposable
    {
        /// <summary>Creates an unfitted model.</summary>
        /// <param name="k">The number of neighbours that vote (scikit-learn's <c>n_neighbors</c>, default 5).</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="k"/> is less than 1.</exception>
        public KnnModel(int k = 5)
        {
            if (k < 1)
                throw new ArgumentOutOfRangeException(nameof(k), k, "At least one neighbour must vote.");
            K = k;
        }

        /// <summary>The number of neighbours that vote.</summary>
        public int K { get; }

        /// <summary>The class labels in ascending order, in the dtype of the training labels (<c>classes_</c>).</summary>
        public NDArray Classes { get; private set; }

        /// <summary>The training samples (a float64 copy), shape (training samples, features).</summary>
        public NDArray TrainingSamples { get; private set; }

        /// <summary>The class index (into <see cref="Classes"/>) of every training sample, int64.</summary>
        public NDArray TrainingCodes { get; private set; }

        /// <summary>The squared norm of every training sample, cached for the distance formula.</summary>
        private NDArray _trainingNorms;

        /// <summary>Stores the training samples and their labels; k-nearest-neighbours has nothing else to learn.</summary>
        /// <param name="X">The training samples, shape (samples, features); any numeric dtype, stored as float64.</param>
        /// <param name="y">One label per sample; any dtype (integers, floats, booleans, text).</param>
        /// <returns>This model, fitted (a previous fit is replaced and its arrays disposed).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array, <paramref name="y"/>'s length differs from the sample
        ///     count, or there are fewer samples than <see cref="K"/>.
        /// </exception>
        public KnnModel Fit(NDArray X, NDArray y)
        {
            // Every temporary of the fit returns to the pool when this scope closes; what the model keeps is
            // detached from it below.
            using var scope = NDScope.Open();
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            if (samples.shape[0] < K)
                throw new ArgumentException($"Expected at least K = {K} samples to fit, got {samples.shape[0]}.", nameof(X));
            NDArray labels = NDHelpers.Targets(y, samples.shape[0], nameof(y));
            var (classes, codes) = NDHelpers.EncodeLabels(labels);
            // Always a copy: the caller may change or dispose X after Fit, and the model reads it at every Predict.
            NDArray training = samples.copy();
            NDArray norms = np.sum(np.square(training), axis: 1);

            NDScope.Detach(new[] { classes, codes, training, norms });
            ReleaseFitted();
            Classes = classes;
            TrainingCodes = codes;
            TrainingSamples = training;
            _trainingNorms = norms;
            return this;
        }

        /// <summary>
        ///     The <see cref="K"/> nearest training samples of every sample, nearest first (<c>kneighbors</c>).
        /// </summary>
        /// <param name="X">The query samples, shape (samples, features) with the training feature count.</param>
        /// <returns>
        ///     <c>distances</c>: float64 (samples, K) Euclidean distances, ascending along each row; <c>indices</c>:
        ///     int64 (samples, K) row numbers into the training samples. The caller owns both.
        /// </returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the training feature count.</exception>
        [NDScoped]
        public (NDArray distances, NDArray indices) KNeighbors(NDArray X)
        {
            NDArray samples = CheckFeatures(X);
            long rows = samples.shape[0];
            NDArray distances = np.empty(new Shape(rows, K));
            NDArray indices = np.empty(new Shape(rows, K), np.int64);
            for (long start = 0; start < rows; start += BlockRows)
            {
                // One query block at a time, in its own scope: its (block, training samples) matrices are returned to
                // the pool before the next block allocates them, so memory stays at one block whatever the batch size.
                using var block = NDScope.Open();
                string range = $"{start}:{Math.Min(start + BlockRows, rows)}";
                NDArray squared = SquaredDistances(samples[range]);
                NDArray nearest = Nearest(squared);
                // argpartition leaves the K nearest unordered: sort just those K per row.
                NDArray nearestSquared = np.take_along_axis(squared, nearest, axis: 1);
                NDArray order = np.argsort(nearestSquared, axis: 1);
                indices[range] = np.take_along_axis(nearest, order, axis: 1);
                distances[range] = np.sqrt(np.take_along_axis(nearestSquared, order, axis: 1));
            }
            return (distances, indices);
        }

        /// <summary>The fraction of each sample's <see cref="K"/> nearest neighbours that carry each class (<c>predict_proba</c>).</summary>
        /// <param name="X">The query samples, shape (samples, features) with the training feature count.</param>
        /// <returns>A float64 (samples, classes) array whose rows sum to 1, columns in <see cref="Classes"/> order.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the training feature count.</exception>
        [NDScoped]
        public NDArray PredictProba(NDArray X) => Votes(CheckFeatures(X)) / K;

        /// <summary>The majority class among every sample's <see cref="K"/> nearest neighbours (<c>predict</c>).</summary>
        /// <param name="X">The query samples, shape (samples, features) with the training feature count.</param>
        /// <returns>A 1-D array of labels in <see cref="Classes"/>' dtype; a vote tie goes to the first class in order.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the training feature count.</exception>
        [NDScoped]
        public NDArray Predict(NDArray X) => Classes[np.argmax(Votes(CheckFeatures(X)), axis: 1)];

        /// <summary>The mean accuracy of <see cref="Predict"/> on labelled samples (<c>score</c>).</summary>
        /// <param name="X">The query samples, shape (samples, features) with the training feature count.</param>
        /// <param name="y">Their true labels, in the dtype of the training labels.</param>
        /// <returns>The fraction of samples whose predicted label equals the true one, in [0, 1].</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array with the training feature count, or <paramref name="y"/>'s
        ///     length differs from the sample count.
        /// </exception>
        [NDScoped]
        public double Score(NDArray X, NDArray y)
        {
            NDArray predicted = Predict(X);
            NDArray truth = NDHelpers.Targets(y, predicted.size, nameof(y));
            return np.mean((predicted == truth).astype(np.float64)).GetDouble();
        }

        /// <summary>Returns the stored training data's buffers to NumSharp's pool; the model is unfitted afterwards.</summary>
        public void Dispose()
        {
            ReleaseFitted();
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///     The number of elements a query block's distance matrix may hold: 4,194,304 float64 (32 MB). Queries are
        ///     processed in blocks of that many distances, so predicting a batch of any size needs a few such matrices,
        ///     where computing all distances at once needed (queries × training samples) several times over (8,000 ×
        ///     8,000 peaked at 3.0 GB).
        /// </summary>
        private const long BlockElements = 1L << 22;

        /// <summary>The query rows per block: as many as fit <see cref="BlockElements"/> distances, at least one.</summary>
        private long BlockRows => Math.Max(1, BlockElements / TrainingSamples.shape[0]);

        /// <summary>The squared Euclidean distance of every query sample to every training sample.</summary>
        /// <param name="samples">Validated float64 query samples (one block).</param>
        /// <returns>A float64 (samples, training samples) array, never negative.</returns>
        [NDScopedCovered]
        private NDArray SquaredDistances(NDArray samples)
        {
            NDArray queryNorms = np.sum(np.square(samples), axis: 1, keepdims: true);
            NDArray dot = np.dot(samples, TrainingSamples.T);
            // |x|^2 - 2 x.t + |t|^2 in one fused pass over the product (one result matrix instead of four
            // temporaries); the expansion can come out a hair below 0 for (near) duplicates, and a distance never is.
            return np.evaluate(NDExpr.Maximum((NDExpr)queryNorms - 2.0 * (NDExpr)dot + _trainingNorms, 0.0));
        }

        /// <summary>The training rows of the <see cref="K"/> smallest distances of every query row, in no particular order.</summary>
        /// <param name="squared">The (samples, training samples) squared distances.</param>
        /// <returns>An int64 (samples, K) array of training row numbers.</returns>
        [NDScopedCovered]
        private NDArray Nearest(NDArray squared)
            => np.argpartition(squared, K - 1, axis: 1)[$":, :{K}"];

        /// <summary>How many of every sample's <see cref="K"/> nearest neighbours carry each class.</summary>
        /// <param name="samples">Validated float64 query samples.</param>
        /// <returns>A float64 (samples, classes) array of vote counts.</returns>
        [NDScopedCovered]
        private NDArray Votes(NDArray samples)
        {
            long rows = samples.shape[0], classes = Classes.size;
            NDArray votes = np.empty(new Shape(rows, classes));
            for (long start = 0; start < rows; start += BlockRows)
            {
                // A scope per block: the block's distances die before the next block allocates its own.
                using var block = NDScope.Open();
                long stop = Math.Min(start + BlockRows, rows);
                NDArray codes = TrainingCodes[Nearest(SquaredDistances(samples[$"{start}:{stop}"]))];   // (block, K)
                // Shift row r's class codes into slots r * classes .. r * classes + classes - 1: one bincount then
                // counts every row's votes at once, without a (block, K, classes) one-hot tensor.
                NDArray slots = codes + (np.arange(stop - start) * classes).reshape(-1, 1);
                NDArray counts = np.bincount(slots.reshape(-1), minlength: (int)((stop - start) * classes));
                votes[$"{start}:{stop}"] = counts.reshape(stop - start, classes).astype(np.float64);
            }
            return votes;
        }

        /// <summary>Validates query input against the fitted model.</summary>
        /// <param name="X">The query samples.</param>
        /// <returns><paramref name="X"/> as float64 (no copy when it already is).</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the training feature count.</exception>
        [NDScopedCovered]
        private NDArray CheckFeatures(NDArray X)
        {
            if (TrainingSamples is null)
                throw new InvalidOperationException("This KnnModel is not fitted yet: call Fit first.");
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            if (samples.shape[1] != TrainingSamples.shape[1])
                throw new ArgumentException($"X has {samples.shape[1]} features, but the model was fitted with {TrainingSamples.shape[1]}.", nameof(X));
            return samples;
        }

        /// <summary>Disposes the arrays of the current fit, if any, and forgets them.</summary>
        private void ReleaseFitted()
        {
            Classes?.Dispose();
            TrainingCodes?.Dispose();
            TrainingSamples?.Dispose();
            _trainingNorms?.Dispose();
            Classes = TrainingCodes = TrainingSamples = _trainingNorms = null;
        }
    }
}
