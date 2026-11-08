using Bigtree.Algorithm.Internal;
using NumSharp;
using System;

namespace Bigtree.Algorithm.LinearRegression
{
    /// <summary>
    ///     Ordinary least squares: the weights <see cref="Coef"/> and offset <see cref="Intercept"/> that minimize the
    ///     squared error of <c>X @ Coef.T + Intercept</c> against the target. scikit-learn's <c>LinearRegression</c>
    ///     contract, computed on NumSharp without a native backend.
    /// </summary>
    /// <remarks>
    ///     <para><b>How it computes.</b> Like scikit-learn, the data are centered first (the intercept is then
    ///     <c>mean(y) - mean(X) @ Coef.T</c>), but the centered problem is solved through the normal equations
    ///     <c>(Xc.T @ Xc) Coef.T = Xc.T @ yc</c> with <c>np.linalg.solve</c>, which NumSharp computes managed (LU).
    ///     scikit-learn calls LAPACK's SVD-based <c>lstsq</c>; NumSharp offers <c>lstsq</c> only with its OpenBLAS
    ///     package installed.</para>
    ///     <para><b>Footgun: conditioning.</b> The normal equations square the condition number of the centered
    ///     <c>X</c>, so on nearly collinear features the coefficients lose about twice as many digits as an SVD solve
    ///     would; exactly collinear features (or more features than samples) make the system singular, and
    ///     <see cref="Fit"/> refuses it instead of returning scikit-learn's minimum-norm solution.</para>
    ///     <para><b>Ownership.</b> The fitted arrays belong to the model; dispose the model (or let the finalizer do it)
    ///     to return their buffers to NumSharp's pool.</para>
    /// </remarks>
    public class LinearRegressionModel : IDisposable
    {
        /// <summary>Creates an unfitted model.</summary>
        /// <param name="fitIntercept">
        ///     <c>true</c> (the default) learns an offset; <c>false</c> forces the fit through the origin and leaves
        ///     <see cref="Intercept"/> at 0.
        /// </param>
        public LinearRegressionModel(bool fitIntercept = true)
        {
            FitIntercept = fitIntercept;
        }

        /// <summary>Whether <see cref="Fit"/> learns an offset.</summary>
        public bool FitIntercept { get; }

        /// <summary>
        ///     The weights (<c>coef_</c>): shape (features,) for a 1-D target, (targets, features) for a 2-D one.
        /// </summary>
        public NDArray Coef { get; private set; }

        /// <summary>
        ///     The offset (<c>intercept_</c>): a 0-d array for a 1-D target, shape (targets,) for a 2-D one; zeros
        ///     when <see cref="FitIntercept"/> is <c>false</c>.
        /// </summary>
        public NDArray Intercept { get; private set; }

        /// <summary>Fits the least-squares weights and offset.</summary>
        /// <param name="X">The samples, shape (samples, features); any numeric dtype, computed as float64.</param>
        /// <param name="y">The target: shape (samples,) or (samples, targets) for several targets at once.</param>
        /// <returns>This model, fitted (a previous fit is replaced and its arrays disposed).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array, or <paramref name="y"/> is not a finite 1-D or 2-D array
        ///     with one row per sample.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     The features are linearly dependent (after centering), so the least-squares solution is not unique.
        /// </exception>
        public LinearRegressionModel Fit(NDArray X, NDArray y)
        {
            // Every temporary of the fit returns to the pool when this scope closes; the fitted arrays are detached
            // from it below, because the model keeps them past this call.
            using var scope = NDScope.Open();
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            NDArray target = Target(y, samples.shape[0]);

            NDArray xOffset = FitIntercept ? np.mean(samples, axis: 0) : np.zeros(new Shape(samples.shape[1]));
            NDArray yOffset = FitIntercept ? np.mean(target, axis: 0) : np.zeros(target.ndim == 1 ? new Shape() : new Shape(target.shape[1]));
            NDArray xc = samples - xOffset, yc = target - yOffset;

            NDArray coef;
            try
            {
                // (features, features) @ coef = (features,) or (features, targets): one LU solve for every target.
                coef = np.linalg.solve(np.dot(xc.T, xc), np.dot(xc.T, yc));
            }
            catch (LinAlgError e)
            {
                // np.linalg.solve raises NumPy's "Singular matrix" for X.T @ X without full rank.
                throw new InvalidOperationException(
                    "The features are linearly dependent (or fewer samples than features), so the least-squares " +
                    "solution is not unique; drop the redundant features.", e);
            }

            // scikit-learn's layout: (features,) for one target, (targets, features) for several.
            coef = target.ndim == 1 ? coef : coef.T.copy();
            NDArray intercept = target.ndim == 1
                ? yOffset - np.dot(xOffset, coef)
                : yOffset - np.dot(xOffset, coef.T);

            NDScope.Detach(new[] { coef, intercept });
            ReleaseFitted();
            Coef = coef;
            Intercept = intercept;
            return this;
        }

        /// <summary>The fitted model's prediction for every sample: <c>X @ Coef.T + Intercept</c>.</summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <returns>A float64 array of shape (samples,) for a 1-D training target, else (samples, targets).</returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the fitted feature count.</exception>
        [NDScoped]
        public NDArray Predict(NDArray X)
        {
            if (Coef is null)
                throw new InvalidOperationException("This LinearRegressionModel is not fitted yet: call Fit first.");
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            long features = Coef.shape[Coef.ndim - 1];
            if (samples.shape[1] != features)
                throw new ArgumentException($"X has {samples.shape[1]} features, but the model was fitted with {features}.", nameof(X));
            return np.dot(samples, Coef.T) + Intercept;
        }

        /// <summary>
        ///     The coefficient of determination R² of the predictions (<c>score</c>): 1 - (residual sum of squares) /
        ///     (total sum of squares), averaged uniformly over the targets of a 2-D target.
        /// </summary>
        /// <param name="X">The samples, shape (samples, features) with the fitted feature count.</param>
        /// <param name="y">Their true targets, shaped like the training target.</param>
        /// <returns>
        ///     1 for a perfect fit, 0 for a model that always predicts the mean, below 0 for a worse one. A constant
        ///     target scores 1 when predicted exactly and 0 otherwise (scikit-learn's <c>r2_score</c> convention).
        /// </returns>
        /// <exception cref="InvalidOperationException">The model is not fitted.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> or <paramref name="y"/> does not match the fitted model.</exception>
        [NDScoped]
        public double Score(NDArray X, NDArray y)
        {
            NDArray predicted = Predict(X);
            NDArray truth = Target(y, predicted.shape[0]);
            if (truth.ndim != predicted.ndim)
                throw new ArgumentException("y must have the shape of the training target.", nameof(y));

            NDArray residual = np.sum(np.square(truth - predicted), axis: 0);
            NDArray total = np.sum(np.square(truth - np.mean(truth, axis: 0)), axis: 0);
            // r2_score: a constant target (total 0) scores 1 if predicted exactly, else 0, instead of dividing by 0.
            NDArray perTarget = np.where(total != 0, 1.0 - residual / np.where(total != 0, total, 1.0),
                                         np.where(residual == 0, 1.0, 0.0));
            return np.mean(perTarget).GetDouble();
        }

        /// <summary>Returns the fitted arrays' buffers to NumSharp's pool; the model is unfitted afterwards.</summary>
        public void Dispose()
        {
            ReleaseFitted();
            GC.SuppressFinalize(this);
        }

        /// <summary>Validates a regression target: finite, 1-D or 2-D, one row per sample; returns it as float64.</summary>
        /// <param name="y">The target.</param>
        /// <param name="rows">The sample count.</param>
        /// <returns><paramref name="y"/> as float64 (no copy when it already is).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="y"/> has the wrong shape or holds NaN or infinity.</exception>
        [NDScopedCovered]
        private static NDArray Target(NDArray y, long rows)
        {
            if (y is null)
                throw new ArgumentNullException(nameof(y));
            if (y.ndim == 2)
            {
                NDArray targets = NDHelpers.Float64Matrix(y, nameof(y));
                return targets.shape[0] == rows
                    ? targets
                    : throw new ArgumentException($"y has {targets.shape[0]} rows for {rows} samples.", nameof(y));
            }

            NDArray target = NDHelpers.Targets(y, rows, nameof(y)).astype(np.float64, copy: false);
            if (!(bool)np.all(np.isfinite(target)))
                throw new ArgumentException("y contains NaN or infinity.", nameof(y));
            return target;
        }

        /// <summary>Disposes the arrays of the current fit, if any, and forgets them.</summary>
        private void ReleaseFitted()
        {
            Coef?.Dispose();
            Intercept?.Dispose();
            Coef = Intercept = null;
        }
    }
}
