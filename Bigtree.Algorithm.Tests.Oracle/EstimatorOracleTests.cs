using System.Linq;
using Bigtree.Algorithm.KNN;
using Bigtree.Algorithm.LinearRegression;
using Bigtree.Algorithm.SVM;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Bigtree.Algorithm.Tests.Oracle
{
    /// <summary>KNN, linear regression and the SVM against scikit-learn 1.8 on the same arrays.</summary>
    [TestClass]
    public class EstimatorOracleTests : OracleTest
    {
        /// <summary>
        ///     KNeighborsClassifier (brute force, k = 7) on random data without distance ties: the neighbour indices,
        ///     the vote fractions and the labels are identical; the distances agree to rounding (both use the
        ///     |x|² - 2 x·t + |t|² expansion, summed in other orders).
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void Knn_MatchesKNeighborsClassifier()
        {
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.neighbors import KNeighborsClassifier\n" +
                    "rng = np.random.default_rng(42)\n" +
                    "X = rng.normal(size=(300, 6)); y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(np.int64) + (X[:, 2] > 0.5)\n" +
                    "Q = rng.normal(size=(80, 6))\n" +
                    "m = KNeighborsClassifier(n_neighbors=7, algorithm='brute').fit(X, y)\n" +
                    "dist, ind = m.kneighbors(Q)\n" +
                    "proba, pred = m.predict_proba(Q), m.predict(Q)");
                NDArray X = py.X, y = py.y, Q = py.Q;
                using var model = new KnnModel(k: 7).Fit(X, y);
                var (distances, indices) = model.KNeighbors(Q);

                AssertEqual(py.ind, indices, "kneighbors indices");
                AssertClose(py.dist, distances, 1e-12, 1e-12, "kneighbors distances");
                AssertEqual(py.proba, model.PredictProba(Q), "predict_proba");
                AssertEqual(py.pred, model.Predict(Q), "predict");
            }
        }

        /// <summary>
        ///     LinearRegression on the diabetes dataset: scikit-learn solves the centered least squares by SVD
        ///     (LAPACK), Bigtree by the normal equations (managed LU); on these well-conditioned features the
        ///     coefficients, intercept, predictions and R² agree to 1e-10.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void LinearRegression_Diabetes()
        {
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.datasets import load_diabetes\n" +
                    "from sklearn.linear_model import LinearRegression\n" +
                    "X, y = load_diabetes(return_X_y=True)\n" +
                    "m = LinearRegression().fit(X, y)\n" +
                    "coef, intercept, pred = m.coef_, np.array(m.intercept_), m.predict(X)\n" +
                    "r2 = np.array(m.score(X, y))");
                NDArray X = py.X, y = py.y;
                using var model = new LinearRegressionModel().Fit(X, y);

                AssertClose(py.coef, model.Coef, 1e-10, 1e-8, "coef_");
                AssertClose(py.intercept, model.Intercept, 1e-12, 0, "intercept_");
                AssertClose(py.pred, model.Predict(X), 1e-12, 1e-9, "predict");
                AssertClose(py.r2, np.array(new[] { model.Score(X, y) }).reshape(new Shape()), 1e-12, 0, "score");
            }
        }

        /// <summary>
        ///     C-SVC with an RBF kernel on iris, both trained on the same [-1, 1]-scaled data: the same support vectors
        ///     (count per class and their indices), the same intercepts (scikit-learn's intercept_ is -rho) and dual
        ///     coefficients, the same one-vs-one decision values and the same labels. libsvm's solver stops at a
        ///     tolerance, but both run it identically, so they agree far below it.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void Svm_Iris_MatchesSVC()
        {
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.datasets import load_iris\n" +
                    "X, y = load_iris(return_X_y=True)\n" +
                    "y = y.astype(np.float64)");
                NDArray X = py.X, y = py.y;

                // Scale with Bigtree's own range transform, then hand the very same matrix to scikit-learn.
                Problem raw = Problem.FromNDArray(X, y);
                Problem scaled = RangeTransform.Compute(raw).Scale(raw);
                var dense = np.array(scaled.X.SelectMany(row => row.Select(node => node.Value)).ToArray()).reshape(150, 4);
                Model model = Training.Train(scaled, new Parameter { KernelType = KernelType.RBF, Gamma = 0.25, C = 2.0 });

                py.Xs = dense;
                py.Exec(
                    "from sklearn.svm import SVC\n" +
                    "m = SVC(kernel='rbf', gamma=0.25, C=2.0, decision_function_shape='ovo').fit(Xs, y)\n" +
                    "n_support, support = m.n_support_.astype(np.int64), np.sort(m.support_).astype(np.int64)\n" +
                    "rho, dual = -m.intercept_, np.sort(np.abs(m.dual_coef_), axis=1)\n" +
                    "dec, pred = m.decision_function(Xs), m.predict(Xs)");

                AssertEqual(py.n_support, np.array(model.NumberOfSVPerClass.Select(n => (long)n).ToArray()), "n_support_");
                AssertEqual(py.support, np.sort(np.array(model.SupportVectorIndices.Select(i => (long)i - 1).ToArray())), "support_");
                AssertClose(py.rho, np.array(model.Rho), 1e-9, 1e-9, "rho (-intercept_)");
                NDArray ours = np.sort(np.abs(np.array(model.SupportVectorCoefficients.SelectMany(row => row).ToArray()).reshape(2, -1)), axis: 1);
                AssertClose(py.dual, ours, 1e-9, 1e-9, "|dual_coef_|");
                AssertClose(py.dec, model.DecisionValues(dense), 1e-9, 1e-9, "decision_function (ovo)");
                AssertEqual(py.pred, model.PredictLabels(dense), "predict");
            }
        }
    }
}
