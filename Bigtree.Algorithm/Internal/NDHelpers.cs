using System;
using NumSharp;

namespace Bigtree.Algorithm.Internal
{
    /// <summary>
    ///     Input checks and small array formulas shared by the NumSharp estimators (Naive Bayes, KNN, linear
    ///     regression, the neural network).
    /// </summary>
    /// <remarks>
    ///     Every member is <see cref="NDScopedCoveredAttribute"/>: it is only called from a scoped estimator member
    ///     (an <c>[NDScoped]</c> method or one that opens an <see cref="NDScope"/> by hand), so the temporaries it
    ///     creates are returned to NumSharp's buffer pool when that caller's scope closes. Calling one of these
    ///     helpers outside a scope still works, but its temporaries then wait for the finalizer.
    /// </remarks>
    internal static class NDHelpers
    {
        /// <summary>
        ///     Validates a feature matrix and returns it as float64, the dtype every estimator computes in.
        /// </summary>
        /// <param name="X">The samples, one row per sample and one column per feature; any numeric dtype.</param>
        /// <param name="paramName">The caller's parameter name, used in exception messages.</param>
        /// <returns>
        ///     <paramref name="X"/> itself when it already is float64 (no copy, so the caller must never write into
        ///     it), otherwise a float64 copy that the caller's scope reclaims.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not 2-D, has no rows or no columns, or holds NaN or infinity (scikit-learn's
        ///     <c>check_array</c> refuses the same inputs, and every estimator here would otherwise return NaN silently).
        /// </exception>
        [NDScopedCovered]
        internal static NDArray Float64Matrix(NDArray X, string paramName)
        {
            if (X is null)
                throw new ArgumentNullException(paramName);
            if (X.ndim != 2)
                throw new ArgumentException($"{paramName} must be a 2-D array (samples, features); got {X.ndim} dimension(s).", paramName);
            if (X.shape[0] == 0 || X.shape[1] == 0)
                throw new ArgumentException($"{paramName} must have at least one sample and one feature; got shape ({X.shape[0]}, {X.shape[1]}).", paramName);

            // copy: false hands back X itself when it already is float64: an estimator reads it, never writes it.
            NDArray values = X.astype(np.float64, copy: false);
            if (!(bool)np.all(np.isfinite(values)))
                throw new ArgumentException($"{paramName} contains NaN or infinity.", paramName);
            return values;
        }

        /// <summary>
        ///     Validates a target vector against the number of samples it labels.
        /// </summary>
        /// <param name="y">One label (or target value) per sample: 1-D, or a single column of shape (n, 1).</param>
        /// <param name="rows">The number of samples in the matching feature matrix.</param>
        /// <param name="paramName">The caller's parameter name, used in exception messages.</param>
        /// <returns>
        ///     A 1-D view of <paramref name="y"/> (a column is flattened the way scikit-learn's <c>column_or_1d</c>
        ///     does); <paramref name="y"/> itself when it already is 1-D.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="y"/> is neither 1-D nor a single column, or its length differs from <paramref name="rows"/>.
        /// </exception>
        [NDScopedCovered]
        internal static NDArray Targets(NDArray y, long rows, string paramName)
        {
            if (y is null)
                throw new ArgumentNullException(paramName);
            NDArray flat = y.ndim == 1 ? y
                : y.ndim == 2 && y.shape[1] == 1 ? y.reshape(-1)
                : throw new ArgumentException($"{paramName} must be 1-D (one label per sample); got {y.ndim} dimension(s).", paramName);
            if (flat.size != rows)
                throw new ArgumentException($"{paramName} has {flat.size} labels for {rows} samples.", paramName);
            return flat;
        }

        /// <summary>
        ///     Encodes labels of any dtype (integers, floats, booleans, text) as class indices, the way scikit-learn's
        ///     <c>LabelEncoder</c> / <c>np.unique</c> does.
        /// </summary>
        /// <param name="y">A 1-D label array.</param>
        /// <returns>
        ///     <c>classes</c>: the distinct labels in ascending order, in <paramref name="y"/>'s own dtype (text labels
        ///     stay text); <c>codes</c>: an int64 array with the index into <c>classes</c> of every label.
        /// </returns>
        [NDScopedCovered]
        internal static (NDArray classes, NDArray codes) EncodeLabels(NDArray y)
        {
            var unique = np.unique(y, return_inverse: true);
            return (unique.values, unique.inverse_indices);
        }

        /// <summary>
        ///     Copies the elements of a 1-D text array (dtype <c>&lt;U…</c> / <c>|S…</c>) into a managed string array.
        /// </summary>
        /// <param name="text">A 1-D NumSharp text array.</param>
        /// <returns>One string per element, in order.</returns>
        /// <remarks>
        ///     NumSharp 0.80.0 has no bulk export for its text dtypes (<c>ToArray&lt;T&gt;</c> requires an unmanaged
        ///     <c>T</c>, and the older <c>AsStringArray</c> decodes the legacy char encoding), so this reads element by
        ///     element through <c>GetValue</c>. Callers use it on label vocabularies, which are short.
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="text"/> is not 1-D.</exception>
        internal static string[] Strings(NDArray text)
        {
            if (text.ndim != 1)
                throw new ArgumentException($"Expected a 1-D text array; got {text.ndim} dimension(s).", nameof(text));
            var strings = new string[text.size];
            for (long i = 0; i < strings.LongLength; i++)
                strings[i] = (string)text.GetValue(i);
            return strings;
        }

        /// <summary>
        ///     Builds a 1-D <c>&lt;U{width}</c> text array from strings, the width (the longest value in UTF-16 units, at
        ///     least 1) measured here and passed as the dtype. Use it instead of <c>np.array(string[])</c> for any
        ///     column of more than a handful of values.
        /// </summary>
        /// <param name="values">The strings, in order.</param>
        /// <returns>A new text array, one element per string, that the caller's scope reclaims.</returns>
        /// <remarks>
        ///     NumSharp 0.80.0's <c>np.array(string[])</c> runs NumPy's dtype discovery on every value (a descriptor per
        ///     value, promoted into the running one): 65-160 ns per string, against 7-18 ns for a sized request, 10 times
        ///     the cost of the copy itself. A UTF-16 width is never smaller than the code points NumPy counts, so no value
        ///     is cut; a surrogate pair only widens the dtype by one.
        /// </remarks>
        /// <exception cref="ArgumentException">
        ///     A value is null: the sized request would store it as the text "None" (and the discovering
        ///     <c>np.array(string[])</c> refuses it with a NotSupportedException about object arrays). Callers that can
        ///     name the row check first, with their own message.
        /// </exception>
        [NDScopedCovered]
        internal static NDArray Text(string[] values)
        {
            int width = 1;
            for (int i = 0; i < values.Length; i++)
                width = Math.Max(width, (values[i] ?? throw new ArgumentException($"Element {i} is null.", nameof(values))).Length);
            return np.array(values, np.dtype($"<U{width}"));
        }

        /// <summary>
        ///     Builds the indicator matrix of encoded labels: row <c>i</c> holds 1.0 in column <c>codes[i]</c> and 0.0
        ///     elsewhere.
        /// </summary>
        /// <param name="codes">An int64 vector of class indices in [0, <paramref name="classes"/>).</param>
        /// <param name="classes">The number of classes (columns).</param>
        /// <returns>A float64 (n, classes) matrix; <c>onehot.T @ X</c> sums the rows of <c>X</c> per class.</returns>
        [NDScopedCovered]
        internal static NDArray OneHot(NDArray codes, long classes)
            => (codes.reshape(-1, 1) == np.arange(classes).reshape(1, -1)).astype(np.float64);

        /// <summary>
        ///     Computes <c>log(sum(exp(a), axis))</c> without overflow, with scikit-learn 1.8's own algorithm
        ///     (<c>sklearn.utils._array_api._logsumexp</c>), so the results match scikit-learn's to the last bit when
        ///     the inputs do.
        /// </summary>
        /// <param name="a">A float64 array.</param>
        /// <param name="axis">The axis to reduce.</param>
        /// <returns><paramref name="a"/>'s shape without <paramref name="axis"/>.</returns>
        /// <remarks>
        ///     The largest entry is taken out of the sum and added back as <c>log1p</c>: <c>max + log(m) + log1p(s / m)</c>,
        ///     where <c>m</c> counts the entries equal to the maximum and <c>s</c> sums <c>exp(a - max)</c> over the rest.
        ///     A slice that is all <c>-inf</c> yields <c>-inf</c> (shift 0), and one holding <c>+inf</c> yields <c>+inf</c>.
        /// </remarks>
        [NDScopedCovered]
        internal static NDArray LogSumExp(NDArray a, int axis)
        {
            NDArray max = np.amax(a, axis: axis, keepdims: true);
            NDArray isMax = a == max;

            // The maxima are summed through m (their count), so they leave the exp-sum: set them to -inf there.
            NDArray rest = a.copy();
            rest[isMax] = double.NegativeInfinity;
            NDArray m = np.sum(isMax.astype(np.float64), axis: axis, keepdims: true);

            // A slice whose maximum is +-inf would turn a - max into NaN; scikit-learn shifts such slices by 0.
            NDArray shift = np.where(np.isfinite(max), max, 0.0);
            NDArray s = np.sum(np.exp(rest - shift), axis: axis, keepdims: true);
            s = np.where(s == 0, s, s / m);
            return np.squeeze(np.log1p(s) + np.log(m) + max, axis);
        }

        /// <summary>The logistic function <c>1 / (1 + exp(-z))</c>, element-wise.</summary>
        /// <param name="z">A float64 array of activations.</param>
        /// <returns>An array of <paramref name="z"/>'s shape with values in (0, 1).</returns>
        [NDScopedCovered]
        internal static NDArray Sigmoid(NDArray z) => 1.0 / (1.0 + np.exp(-z));
    }
}
