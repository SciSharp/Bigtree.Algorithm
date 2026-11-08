using Bigtree.Algorithm.LinearRegression;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;

namespace Bigtree.Algorithm.UnitTest.LinearRegression
{
    /// <summary>
    ///     Ordinary least squares: exact recovery, the residual geometry of a noisy fit, several targets, no
    ///     intercept, collinear features. (Bigtree 0.4.0's test only plotted points through MatplotlibCS from a path on
    ///     its author's disk; the model class was empty.)
    /// </summary>
    [TestClass]
    public class ModelTest
    {
        /// <summary>A target built exactly from known weights is recovered, and R² is 1.</summary>
        [TestMethod]
        public void RecoversExactWeights()
        {
            using var scope = NDScope.Open();
            np.random.seed(11);
            var X = np.random.uniform(-1.0, 1.0, new Shape(40, 3));
            var weights = np.array(new[] { 1.5, -2.0, 0.5 });
            var y = np.dot(X, weights) + 4.0;
            using var model = new LinearRegressionModel().Fit(X, y);

            double[] coef = model.Coef.ToArray<double>();
            for (int i = 0; i < 3; i++)
                Assert.AreEqual(weights.GetDouble(i), coef[i], 1e-10);
            Assert.AreEqual(0, model.Intercept.ndim, "one target gives a scalar intercept, as scikit-learn's float");
            Assert.AreEqual(4.0, model.Intercept.GetDouble(), 1e-10);
            Assert.AreEqual(1.0, model.Score(X, y), 1e-12);
        }

        /// <summary>
        ///     The noisy curve of 0.4.0's test (sin(x) + 0.1 x² + noise): regressed on [x, x²], the residuals are
        ///     orthogonal to every feature and sum to 0, the defining property of least squares.
        /// </summary>
        [TestMethod]
        public void Regression()
        {
            using var scope = NDScope.Open();
            np.random.seed(0);
            var data_x = np.linspace(1, 10, 100);
            var data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100);
            var features = np.stack(new[] { data_x, np.power(data_x, 2) }, axis: 1);
            using var model = new LinearRegressionModel().Fit(features, data_y);

            NDArray residual = data_y - model.Predict(features);
            Assert.AreEqual(0.0, np.sum(residual).GetDouble(), 1e-9);
            Assert.AreEqual(0.0, np.dot(residual, features[":, 0"]).GetDouble(), 1e-8);
            Assert.AreEqual(0.0, np.dot(residual, features[":, 1"]).GetDouble(), 1e-7);
            Assert.IsTrue(model.Score(features, data_y) > 0.8);
        }

        /// <summary>Two targets at once give (targets, features) weights and one intercept per target.</summary>
        [TestMethod]
        public void SeveralTargets()
        {
            using var scope = NDScope.Open();
            np.random.seed(2);
            var X = np.random.randn(30, 2);
            var Y = np.stack(new[] { np.dot(X, np.array(new[] { 1.0, 2.0 })) + 1.0, np.dot(X, np.array(new[] { -3.0, 0.5 })) - 2.0 }, axis: 1);
            using var model = new LinearRegressionModel().Fit(X, Y);

            CollectionAssert.AreEqual(new long[] { 2, 2 }, model.Coef.shape);
            double[] coef = model.Coef.ToArray<double>();
            double[] expected = { 1.0, 2.0, -3.0, 0.5 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], coef[i], 1e-10);
            Assert.AreEqual(1.0, model.Intercept.GetDouble(0), 1e-10);
            Assert.AreEqual(-2.0, model.Intercept.GetDouble(1), 1e-10);
            CollectionAssert.AreEqual(new long[] { 30, 2 }, model.Predict(X).shape);
        }

        /// <summary>Without an intercept the fit goes through the origin.</summary>
        [TestMethod]
        public void WithoutIntercept()
        {
            using var scope = NDScope.Open();
            var X = np.array(new double[,] { { 1 }, { 2 }, { 3 } });
            var y = np.array(new[] { 2.0, 4.0, 7.0 });
            using var model = new LinearRegressionModel(fitIntercept: false).Fit(X, y);
            // argmin |Xw - y|^2 = (1*2 + 2*4 + 3*7) / (1 + 4 + 9) = 31 / 14
            Assert.AreEqual(31.0 / 14.0, model.Coef.GetDouble(0), 1e-15);
            Assert.AreEqual(0.0, model.Intercept.GetDouble());
        }

        /// <summary>
        ///     A fit whose centered rows span several blocks (60,000 × 80 = 4.8 M elements against blocks of 2^22, so
        ///     52,428 + 7,572 rows) agrees with the normal equations solved over the whole centered matrix, for two
        ///     targets at once, and recovers the generating weights through the noise.
        /// </summary>
        [TestMethod]
        public void FitAcrossSeveralBlocksMatchesTheWholeProduct()
        {
            using var scope = NDScope.Open();
            var rng = np.random.default_rng(21);
            NDArray X = rng.normal(size: new Shape(60000, 80)) + 3.0;
            NDArray weights = rng.normal(size: new Shape(80, 2));
            NDArray y = np.dot(X, weights) + np.array(new[] { 1.0, -2.0 }) + 0.01 * rng.normal(size: new Shape(60000, 2));
            using var model = new LinearRegressionModel().Fit(X, y);

            NDArray xc = X - np.mean(X, axis: 0), yc = y - np.mean(y, axis: 0);
            double[] whole = np.linalg.solve(np.dot(xc.T, xc), np.dot(xc.T, yc)).T.copy().ToArray<double>();
            double[] blocked = model.Coef.ToArray<double>();
            double[] generating = weights.T.copy().ToArray<double>();
            for (int i = 0; i < whole.Length; i++)
            {
                Assert.AreEqual(whole[i], blocked[i], 1e-12, $"coefficient {i}");
                Assert.AreEqual(generating[i], blocked[i], 1e-3, $"coefficient {i} against the generating weight");
            }
        }

        /// <summary>Collinear features have no unique solution and are refused; a constant target scores 1 when hit exactly.</summary>
        [TestMethod]
        public void CollinearFeaturesAreRefused()
        {
            using var scope = NDScope.Open();
            var x = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            var X = np.stack(new[] { x, 2 * x }, axis: 1);
            Assert.ThrowsException<InvalidOperationException>(() => new LinearRegressionModel().Fit(X, x));

            using var constant = new LinearRegressionModel().Fit(x.reshape(-1, 1), np.full(new Shape(4), 3.0));
            Assert.AreEqual(1.0, constant.Score(x.reshape(-1, 1), np.full(new Shape(4), 3.0)));
        }
    }
}
