using Bigtree.Algorithm.Internal;
using NumSharp;
using System;

namespace Bigtree.Algorithm.Bayes
{
    /// <summary>
    ///     Bernoulli Naive Bayes for binary (present / absent) features: every feature is a coin whose bias depends
    ///     on the class. scikit-learn's <c>BernoulliNB</c>, computed on NumSharp.
    /// </summary>
    /// <remarks>
    ///     <para><b>How it computes.</b> The features are binarized (<c>X &gt; binarize</c>), the per-class feature
    ///     counts are one matrix product <c>onehot(y).T @ X</c>, and the joint log-likelihood of a batch is one more:
    ///     <c>X @ (log p - log(1 - p)).T + log prior + sum(log(1 - p))</c>, which scores absent features without
    ///     materializing <c>1 - X</c>. The formulas are scikit-learn 1.8's; its matrix products run on BLAS and
    ///     NumSharp's on its own kernels, which sum in a different order, so results agree to about 1e-12 rather
    ///     than bit for bit.</para>
    ///     <para><b>Ownership.</b> The fitted arrays belong to the model: never dispose them. Dispose the model (or let
    ///     the finalizer do it) to return their buffers to NumSharp's pool.</para>
    ///     <para>Not thread-safe while fitting; concurrent predictions on a fitted model are safe.</para>
    /// </remarks>
    public class BernoulliNaiveBayes : IDisposable
    {
        /// <summary>Creates an unfitted model.</summary>
        /// <param name="alpha">
        ///     Additive (Laplace / Lidstone) smoothing added to every feature count; 0 means none, in which case a
        ///     feature never seen in a class gives that class probability 0.
        /// </param>
        /// <param name="binarize">
        ///     The threshold that maps features to {0, 1}: a value above it is 1. <c>null</c> assumes the input is
        ///     already binary and uses it as is.
        /// </param>
        /// <param name="fitPrior">
        ///     <c>true</c> learns the class priors from the class frequencies; <c>false</c> uses a uniform prior.
        ///     Ignored when <paramref name="classPrior"/> is given.
        /// </param>
        /// <param name="classPrior">Fixed class priors, one per class in ascending label order; the model keeps a copy.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="alpha"/> is negative or NaN.</exception>
        public BernoulliNaiveBayes(double alpha = 1.0, double? binarize = 0.0, bool fitPrior = true, NDArray classPrior = null)
        {
            if (!(alpha >= 0))
                throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "The smoothing alpha must be 0 or more.");
            Alpha = alpha;
            Binarize = binarize;
            FitPrior = fitPrior;
            if (classPrior is not null)
            {
                ClassPriorGiven = classPrior.astype(np.float64);
                // The model owns its copy, even when it is constructed inside a caller's scope.
                NDScope.Detach(ClassPriorGiven);
            }
        }

        /// <summary>The additive smoothing added to every feature count.</summary>
        public double Alpha { get; }

        /// <summary>The binarization threshold (a value above it is 1), or <c>null</c> when the input is used as is.</summary>
        public double? Binarize { get; }

        /// <summary>Whether the class priors are learned from the class frequencies (when no fixed prior is given).</summary>
        public bool FitPrior { get; }

        /// <summary>The fixed class priors given at construction (a copy), or <c>null</c>.</summary>
        public NDArray ClassPriorGiven { get; private set; }

        /// <summary>The class labels in ascending order, in the dtype of the training labels (<c>classes_</c>).</summary>
        public NDArray Classes { get; private set; }

        /// <summary>The number of training samples of each class, as float64 (<c>class_count_</c>).</summary>
        public NDArray ClassCount { get; private set; }

        /// <summary>The number of training samples of each class that have each feature, shape (classes, features) (<c>feature_count_</c>).</summary>
        public NDArray FeatureCount { get; private set; }

        /// <summary>The smoothed log-probability that a feature is present given the class, shape (classes, features) (<c>feature_log_prob_</c>).</summary>
        public NDArray FeatureLogProb { get; private set; }

        /// <summary>The log prior of each class (<c>class_log_prior_</c>).</summary>
        public NDArray ClassLogPrior { get; private set; }

        /// <summary>Counts the features per class and turns the counts into smoothed log-probabilities.</summary>
        /// <param name="X">The training samples, shape (samples, features); binarized with <see cref="Binarize"/>.</param>
        /// <param name="y">One label per sample; any dtype (integers, floats, booleans, text).</param>
        /// <returns>This model, fitted (a previous fit is replaced and its arrays disposed).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array, <paramref name="y"/>'s length differs from the sample
        ///     count, or the fixed priors do not have one entry per class.
        /// </exception>
        public BernoulliNaiveBayes Fit(NDArray X, NDArray y)
        {
            // Every temporary of the fit returns to the pool when this scope closes; the fitted arrays are detached
            // from it below, because the model keeps them past this call.
            using var scope = NDScope.Open();
            NDArray samples = Binarized(NDHelpers.Float64Matrix(X, nameof(X)));
            NDArray labels = NDHelpers.Targets(y, samples.shape[0], nameof(y));
            var (classes, codes) = NDHelpers.EncodeLabels(labels);

            NDArray onehot = NDHelpers.OneHot(codes, classes.size);
            NDArray featureCount = np.dot(onehot.T, samples);
            NDArray classCount = np.sum(onehot, axis: 0);
            // scikit-learn's _update_feature_log_prob: each feature has two outcomes, so the class count is
            // smoothed by 2 * alpha.
            NDArray featureLogProb = np.log(featureCount + Alpha) - np.log((classCount + Alpha * 2).reshape(-1, 1));
            NDArray classLogPrior = LogPrior(classCount);

            NDScope.Detach(new[] { classes, classCount, featureCount, featureLogProb, classLogPrior });
            ReleaseFitted();
            Classes = classes;
            ClassCount = classCount;
            FeatureCount = featureCount;
            FeatureLogProb = featureLogProb;
            ClassLogPrior = classLogPrior;
            return this;
        }

        /// <summary>The joint log-likelihood <c>log P(c) + log P(x | c)</c> of every sample and class (<c>predict_joint_log_proba</c>).</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count; binarized like the training data.</param>
        /// <returns>A float64 (samples, classes) array, columns in <see cref="Classes"/> order.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray PredictJointLogProba(NDArray X)
        {
            NDArray samples = CheckFeatures(X);
            // log(1 - p) of every (class, feature): the score of an ABSENT feature.
            NDArray negProb = np.log(1 - np.exp(FeatureLogProb));
            // sum_x [x log p + (1 - x) log(1 - p)] = x . (log p - log(1 - p)) + sum log(1 - p), so the absent
            // features never need a (samples, features) array of their own.
            NDArray jll = np.dot(samples, (FeatureLogProb - negProb).T);
            return jll + (ClassLogPrior + np.sum(negProb, axis: 1));
        }

        /// <summary>The log of the class probabilities of every sample (<c>predict_log_proba</c>).</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <returns>A float64 (samples, classes) array whose rows log-sum-exp to 0.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray PredictLogProba(NDArray X)
        {
            NDArray jll = PredictJointLogProba(X);
            return jll - np.atleast_2d(NDHelpers.LogSumExp(jll, 1)).T;
        }

        /// <summary>The class probabilities of every sample (<c>predict_proba</c>).</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <returns>A float64 (samples, classes) array whose rows sum to 1.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray PredictProba(NDArray X) => np.exp(PredictLogProba(X));

        /// <summary>The most probable class of every sample (<c>predict</c>).</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <returns>A 1-D array of labels in <see cref="Classes"/>' dtype; a tie goes to the first class in order.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray Predict(NDArray X) => Classes[np.argmax(PredictJointLogProba(X), axis: 1)];

        /// <summary>The mean accuracy of <see cref="Predict"/> on labelled samples (<c>score</c>).</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <param name="y">Their true labels, in the dtype of the training labels.</param>
        /// <returns>The fraction of samples whose predicted label equals the true one, in [0, 1].</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array with the fitted feature count, or <paramref name="y"/>'s
        ///     length differs from the sample count.
        /// </exception>
        [NDScoped]
        public double Score(NDArray X, NDArray y)
        {
            NDArray predicted = Predict(X);
            NDArray truth = NDHelpers.Targets(y, predicted.size, nameof(y));
            return np.mean((predicted == truth).astype(np.float64)).GetDouble();
        }

        /// <summary>Returns the fitted arrays' buffers to NumSharp's pool; the model is unfitted afterwards.</summary>
        public void Dispose()
        {
            ReleaseFitted();
            ClassPriorGiven?.Dispose();
            ClassPriorGiven = null;
            GC.SuppressFinalize(this);
        }

        /// <summary>Maps features to {0, 1} with <see cref="Binarize"/> (scikit-learn's <c>binarize</c>), or passes them through.</summary>
        /// <param name="samples">A float64 sample matrix.</param>
        /// <returns>A float64 array of 0.0 and 1.0, or <paramref name="samples"/> itself when <see cref="Binarize"/> is <c>null</c>.</returns>
        [NDScopedCovered]
        private NDArray Binarized(NDArray samples)
            => Binarize is double threshold ? (samples > threshold).astype(np.float64) : samples;

        /// <summary>
        ///     The class log priors (scikit-learn's <c>_update_class_log_prior</c>): the fixed priors when given,
        ///     else the log class frequencies, else uniform.
        /// </summary>
        /// <param name="classCount">The number of training samples of each class.</param>
        /// <returns>A float64 vector with one log prior per class.</returns>
        /// <exception cref="ArgumentException">The fixed priors do not have one entry per class.</exception>
        [NDScopedCovered]
        private NDArray LogPrior(NDArray classCount)
        {
            long classes = classCount.size;
            if (ClassPriorGiven is not null)
            {
                if (ClassPriorGiven.size != classes)
                    throw new ArgumentException("Number of priors must match number of classes.", "classPrior");
                return np.log(ClassPriorGiven);
            }

            return FitPrior
                ? np.log(classCount) - np.log(np.sum(classCount))
                : np.full(new Shape(classes), -Math.Log(classes));
        }

        /// <summary>Validates prediction input against the fitted model and binarizes it like the training data.</summary>
        /// <param name="X">The samples to predict.</param>
        /// <returns>The binarized float64 samples.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScopedCovered]
        private NDArray CheckFeatures(NDArray X)
        {
            if (FeatureLogProb is null)
                throw new InvalidOperationException("This BernoulliNaiveBayes is not fitted yet: call Fit first.");
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            if (samples.shape[1] != FeatureLogProb.shape[1])
                throw new ArgumentException($"X has {samples.shape[1]} features, but the model was fitted with {FeatureLogProb.shape[1]}.", nameof(X));
            return Binarized(samples);
        }

        /// <summary>Disposes the arrays of the current fit, if any, and forgets them.</summary>
        private void ReleaseFitted()
        {
            Classes?.Dispose();
            ClassCount?.Dispose();
            FeatureCount?.Dispose();
            FeatureLogProb?.Dispose();
            ClassLogPrior?.Dispose();
            Classes = ClassCount = FeatureCount = FeatureLogProb = ClassLogPrior = null;
        }
    }
}
