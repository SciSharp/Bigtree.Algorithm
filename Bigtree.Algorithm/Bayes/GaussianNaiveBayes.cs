using Bigtree.Algorithm.Internal;
using NumSharp;
using NumSharp.Backends.Iteration;
using System;

namespace Bigtree.Algorithm.Bayes
{
    /// <summary>
    ///     Gaussian Naive Bayes: each feature is modelled, per class, as an independent normal distribution with the
    ///     class's mean and variance. scikit-learn's <c>GaussianNB</c>, computed on NumSharp.
    /// </summary>
    /// <remarks>
    ///     <para><b>Same numbers as scikit-learn.</b> Fitting and prediction follow <c>sklearn.naive_bayes.GaussianNB</c>
    ///     (1.8) operation for operation: the per-class <c>np.mean</c> / <c>np.var</c>, the variance smoothing
    ///     <c>epsilon = var_smoothing * max(var(X, axis=0))</c>, the per-class joint log-likelihood loop, and
    ///     scikit-learn's own log-sum-exp for the normalization. NumSharp's reductions and ufuncs are bit-compatible
    ///     with NumPy's, so the fitted attributes and the probabilities match scikit-learn's to the last bit on the
    ///     same input (the oracle test project checks exactly that).</para>
    ///     <para><b>Ownership.</b> The fitted arrays (<see cref="Classes"/>, <see cref="Theta"/>, ...) belong to the
    ///     model: never dispose them, and do not use them after disposing the model. Dispose the model (or let the
    ///     finalizer do it) to return their buffers to NumSharp's pool. The arrays a prediction returns belong to the
    ///     caller.</para>
    ///     <para>Not thread-safe while fitting; concurrent predictions on a fitted model are safe.</para>
    /// </remarks>
    public class GaussianNaiveBayes : IDisposable
    {
        /// <summary>
        ///     scikit-learn's <c>((X - theta) ** 2) / var</c> as one positional float64 program compiled once: Input(0)
        ///     the samples (samples, features), Input(1) a class's means and Input(2) its variances (features,), which
        ///     broadcast over the rows as the ufuncs' operands did.
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>** 2</c> is its square (a multiply), so the program's subtract, square and divide are
        ///     scikit-learn's operations in its order and the bits stay scikit-learn's. Immutable and safe to evaluate
        ///     from several threads, as concurrent predictions do.
        /// </remarks>
        private static readonly CompiledExpression WeightedSquare =
            (NDExpr.Square(NDExpr.Input(0) - NDExpr.Input(1)) / NDExpr.Input(2)).Compile(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double);

        /// <summary>Creates an unfitted model.</summary>
        /// <param name="priors">
        ///     Fixed class priors, one per class in ascending label order, summing to 1. <c>null</c> (the default)
        ///     estimates them from the class frequencies at fit time. The model keeps its own copy.
        /// </param>
        /// <param name="varSmoothing">
        ///     The fraction of the largest feature variance added to every variance, so a feature that is constant
        ///     within a class does not divide by zero. scikit-learn's default 1e-9.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="varSmoothing"/> is negative or NaN.</exception>
        public GaussianNaiveBayes(NDArray priors = null, double varSmoothing = 1e-9)
        {
            if (!(varSmoothing >= 0))
                throw new ArgumentOutOfRangeException(nameof(varSmoothing), varSmoothing, "The variance smoothing must be 0 or more.");
            VarSmoothing = varSmoothing;
            if (priors is not null)
            {
                Priors = priors.astype(np.float64);
                // The model owns its copy, even when it is constructed inside a caller's scope.
                NDScope.Detach(Priors);
            }
        }

        /// <summary>The fixed class priors given at construction (a copy), or <c>null</c> when they are estimated.</summary>
        public NDArray Priors { get; private set; }

        /// <summary>The fraction of the largest feature variance added to every variance at fit time.</summary>
        public double VarSmoothing { get; }

        /// <summary>The class labels in ascending order, in the dtype of the <c>y</c> passed to <see cref="Fit"/> (<c>classes_</c>).</summary>
        public NDArray Classes { get; private set; }

        /// <summary>The number of training samples of each class, as float64 (<c>class_count_</c>).</summary>
        public NDArray ClassCount { get; private set; }

        /// <summary>The probability of each class (<c>class_prior_</c>): <see cref="Priors"/> when given, else the class frequencies.</summary>
        public NDArray ClassPrior { get; private set; }

        /// <summary>The mean of each feature per class, shape (classes, features) (<c>theta_</c>).</summary>
        public NDArray Theta { get; private set; }

        /// <summary>The variance of each feature per class plus <see cref="Epsilon"/>, shape (classes, features) (<c>var_</c>).</summary>
        public NDArray Var { get; private set; }

        /// <summary>The absolute amount added to every variance: <see cref="VarSmoothing"/> times the largest feature variance (<c>epsilon_</c>).</summary>
        public double Epsilon { get; private set; }

        /// <summary>
        ///     Estimates the class priors and the per-class mean and variance of every feature.
        /// </summary>
        /// <param name="X">The training samples, shape (samples, features); any numeric dtype, computed as float64.</param>
        /// <param name="y">One label per sample; any dtype (integers, floats, booleans, text).</param>
        /// <returns>This model, fitted (a previous fit is replaced and its arrays disposed).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array, <paramref name="y"/>'s length differs from the sample
        ///     count, or the constructor's priors do not fit the classes (count, sum 1, non-negative).
        /// </exception>
        public GaussianNaiveBayes Fit(NDArray X, NDArray y)
        {
            // Every temporary of the fit returns to the pool when this scope closes; the fitted arrays are detached
            // from it below, because the model keeps them past this call.
            using var scope = NDScope.Open();
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            NDArray labels = NDHelpers.Targets(y, samples.shape[0], nameof(y));
            var (classes, codes) = NDHelpers.EncodeLabels(labels);
            long classCount = classes.size;

            double epsilon = VarSmoothing * np.amax(np.var(samples, axis: 0)).GetDouble();

            // One row of theta / var per class, in ascending class order, exactly like scikit-learn's _partial_fit.
            var means = new NDArray[classCount];
            var variances = new NDArray[classCount];
            var counts = new double[classCount];
            for (int i = 0; i < classCount; i++)
            {
                // A scope per class: the copy of the class's rows (and the mask) is freed before the next class's,
                // instead of every class's copy living until the fit returns.
                using var perClass = NDScope.Open();
                NDArray members = samples[codes == i];
                means[i] = perClass.Returns(np.mean(members, axis: 0));
                variances[i] = perClass.Returns(np.var(members, axis: 0));
                counts[i] = members.shape[0];
            }

            NDArray theta = np.stack(means);
            NDArray variance = np.stack(variances) + epsilon;
            NDArray count = np.array(counts);
            NDArray prior = Priors is null ? count / np.sum(count) : CheckPriors(Priors, classCount);

            NDScope.Detach(new[] { classes, count, prior, theta, variance });
            ReleaseFitted();
            Classes = classes;
            ClassCount = count;
            ClassPrior = prior;
            Theta = theta;
            Var = variance;
            Epsilon = epsilon;
            return this;
        }

        /// <summary>
        ///     The joint log-likelihood <c>log P(c) + log P(x | c)</c> of every sample and class, before normalization
        ///     (<c>predict_joint_log_proba</c>).
        /// </summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <returns>A float64 (samples, classes) array, columns in <see cref="Classes"/> order.</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray PredictJointLogProba(NDArray X)
        {
            NDArray samples = CheckFeatures(X);
            NDArray logPrior = np.log(ClassPrior);
            var columns = new NDArray[Classes.size];
            for (int i = 0; i < columns.Length; i++)
            {
                // A scope per class: its (samples, features) temporary is freed before the next class allocates it
                // again. In the method's one scope all classes' temporaries stayed alive until it returned: 100,000 x 50
                // with 10 classes peaked at 1.4 GB for a 40 MB input.
                using var perClass = NDScope.Open();
                NDArray theta = Theta[i], variance = Var[i];
                // scikit-learn's _joint_log_likelihood, term for term: a scalar normalizer, minus half the
                // variance-weighted squared distance of every sample, plus the log prior. The distance's element-wise
                // part, ((X - theta) ** 2) / var, is one fused pass (WeightedSquare): the same three operations in the
                // same order, one temporary instead of three.
                NDArray normalizer = -0.5 * np.sum(np.log(2.0 * Math.PI * variance));
                NDArray distance = normalizer - 0.5 * np.sum(WeightedSquare.Evaluate(new[] { samples, theta, variance }), axis: 1);
                columns[i] = perClass.Returns(logPrior[i] + distance);
            }

            // scikit-learn stacks the per-class columns as rows and transposes: the (samples, classes) result is a
            // column-major view. Keeping that layout keeps the reduction order of a later log-sum-exp identical.
            return np.stack(columns).T;
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
            NDArray evidence = NDHelpers.LogSumExp(jll, 1);
            return jll - np.atleast_2d(evidence).T;
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
            Priors?.Dispose();
            Priors = null;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///     Checks the constructor's priors against the fitted classes, the way scikit-learn's <c>_partial_fit</c>
        ///     does, and copies them so the fitted <see cref="ClassPrior"/> has its own buffer.
        /// </summary>
        /// <param name="priors">The priors given at construction.</param>
        /// <param name="classCount">The number of classes found in <c>y</c>.</param>
        /// <returns>A float64 copy of <paramref name="priors"/>.</returns>
        /// <exception cref="ArgumentException">The count differs from the class count, the sum is not 1, or a prior is negative.</exception>
        [NDScopedCovered]
        private static NDArray CheckPriors(NDArray priors, long classCount)
        {
            if (priors.ndim != 1 || priors.size != classCount)
                throw new ArgumentException("Number of priors must match number of classes.", nameof(priors));
            if (!(bool)np.isclose(np.sum(priors), 1.0))
                throw new ArgumentException("The sum of the priors should be 1.", nameof(priors));
            if ((bool)np.any(priors < 0))
                throw new ArgumentException("Priors must be non-negative.", nameof(priors));
            return priors.copy();
        }

        /// <summary>
        ///     Validates prediction input against the fitted model.
        /// </summary>
        /// <param name="X">The samples to predict.</param>
        /// <returns><paramref name="X"/> as float64 (no copy when it already is).</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScopedCovered]
        private NDArray CheckFeatures(NDArray X)
        {
            if (Theta is null)
                throw new InvalidOperationException("This GaussianNaiveBayes is not fitted yet: call Fit first.");
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            if (samples.shape[1] != Theta.shape[1])
                throw new ArgumentException($"X has {samples.shape[1]} features, but the model was fitted with {Theta.shape[1]}.", nameof(X));
            return samples;
        }

        /// <summary>Disposes the arrays of the current fit, if any, and forgets them.</summary>
        private void ReleaseFitted()
        {
            Classes?.Dispose();
            ClassCount?.Dispose();
            ClassPrior?.Dispose();
            Theta?.Dispose();
            Var?.Dispose();
            Classes = ClassCount = ClassPrior = Theta = Var = null;
        }
    }
}
