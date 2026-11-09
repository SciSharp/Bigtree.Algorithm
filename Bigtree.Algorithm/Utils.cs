using Bigtree.Algorithm.Internal;
using NumSharp;
using System;

namespace Bigtree.Algorithm
{
    /// <summary>Dataset helpers: reading a labelled CSV, cross-validation folds and feature scaling, on NumSharp.</summary>
    public class Utils
    {
        /// <summary>
        ///     Reads a comma-separated file of numbers whose last column is a class label (the layout of the UCI seeds
        ///     dataset) into a feature matrix and class indices.
        /// </summary>
        /// <param name="path">The file: one sample per line, no header; blank lines and <c>#</c> comments are skipped.</param>
        /// <returns>
        ///     <c>X</c>: a float64 (samples, features) matrix of every column but the last. <c>y</c>: an int32 vector
        ///     numbering the distinct labels of the last column 0, 1, ... in ascending order of their values (so the
        ///     seeds labels 1, 2, 3 become 0, 1, 2). The caller owns both.
        /// </returns>
        /// <remarks>
        ///     One <c>np.loadtxt</c> pass parses the whole file, culture-invariantly, and <c>np.unique</c> encodes the
        ///     labels. Bigtree 0.4.0 numbered the labels in order of first appearance and stopped at the first blank
        ///     line; the numbering is the same whenever the file lists its classes in ascending order.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="path"/> is <c>null</c>.</exception>
        /// <exception cref="System.IO.IOException">The file cannot be read.</exception>
        /// <exception cref="FormatException">A field is not a number, or the rows differ in length (raised by <c>np.loadtxt</c>).</exception>
        [NDScoped]
        public static (NDArray, NDArray) ReadCsv(string path)
        {
            if (path is null)
                throw new ArgumentNullException(nameof(path));
            NDArray data = np.loadtxt(path, delimiter: ",", ndmin: 2);
            // A copy, so X is a dense matrix of its own rather than a view that keeps the label column alive.
            NDArray X = data[":, :-1"].copy();
            NDArray y = NDHelpers.EncodeLabels(data[":, -1"]).codes.astype(np.int32);
            return (X, y);
        }

        /// <summary>
        ///     Splits the indices 0 .. <paramref name="N"/> - 1 into <paramref name="n_folds"/> disjoint random folds of
        ///     <c>N / n_folds</c> indices each, for k-fold cross-validation (NN-scratch's <c>crossval_folds</c>).
        /// </summary>
        /// <param name="N">The number of samples.</param>
        /// <param name="n_folds">The number of folds, between 1 and <paramref name="N"/>.</param>
        /// <param name="seed">
        ///     The seed of the NumPy-compatible generator: the permutation equals NumPy's
        ///     <c>default_rng(seed).permutation(N)</c>. <c>null</c> seeds from the operating system.
        /// </param>
        /// <returns>
        ///     One int64 index vector per fold. When <paramref name="N"/> is not a multiple of
        ///     <paramref name="n_folds"/>, the last <c>N % n_folds</c> indices of the permutation belong to no fold
        ///     (as in NN-scratch), so every fold has the same size.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="n_folds"/> is not in [1, <paramref name="N"/>].</exception>
        [NDScoped]
        public static NDArray[] CrossValFolds(int N, int n_folds, int? seed = null)
        {
            if (n_folds < 1 || n_folds > N)
                throw new ArgumentOutOfRangeException(nameof(n_folds), n_folds, $"The fold count must be between 1 and N ({N}).");

            var generator = seed is int value ? np.random.default_rng(value) : np.random.default_rng();
            NDArray permutation = generator.permutation(N);
            int size = N / n_folds;
            var folds = new NDArray[n_folds];
            for (int i = 0; i < n_folds; i++)
                folds[i] = permutation[$"{i * size}:{(i + 1) * size}"].copy();
            // The woven scope hands the returned folds to the caller's scope (if any), which owns them; the
            // permutation they were copied from is reclaimed.
            return folds;
        }

        /// <summary>
        ///     Scales every column into [0, 1]: <c>(X - min) / (max - min)</c> per feature (min-max normalization, what
        ///     NumSharp 0.30's in-place <c>NDArray.Normalize()</c> did).
        /// </summary>
        /// <param name="X">The samples, shape (samples, features); any numeric dtype, computed as float64.</param>
        /// <returns>
        ///     A new float64 array; the caller owns it. A constant column has no range and becomes NaN, as in NumPy.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array.</exception>
        [NDScoped]
        public static NDArray Normalize(NDArray X)
        {
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            NDArray min = np.amin(samples, axis: 0);
            return (samples - min) / (np.amax(samples, axis: 0) - min);
        }
    }
}
