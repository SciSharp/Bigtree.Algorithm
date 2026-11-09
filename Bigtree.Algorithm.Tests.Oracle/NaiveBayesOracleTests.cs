using System;
using System.Collections.Generic;
using System.Linq;
using Bigtree.Algorithm.Bayes;
using Bigtree.Algorithm.Statistics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Bigtree.Algorithm.Tests.Oracle
{
    /// <summary>
    ///     Naive Bayes against scikit-learn 1.8 on its own datasets: Gaussian to the last bit, Bernoulli to BLAS
    ///     rounding, and Bigtree's Multinomial formula against a NumPy evaluation of it.
    /// </summary>
    [TestClass]
    public class NaiveBayesOracleTests : OracleTest
    {
        /// <summary>
        ///     GaussianNB on iris (train / test split): every fitted attribute and every prediction output is
        ///     bit-identical to scikit-learn's, because GaussianNaiveBayes repeats its operations and NumSharp's
        ///     reductions and ufuncs are NumPy's.
        /// </summary>
        [TestMethod]
        public void Gaussian_Iris_BitIdentical()
        {
            using var scope = NDScope.Open();
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.datasets import load_iris\n" +
                    "from sklearn.naive_bayes import GaussianNB\n" +
                    "X, y = load_iris(return_X_y=True)\n" +
                    "Xtr, ytr, Xte = X[::2].copy(), y[::2].copy(), X[1::2].copy()\n" +
                    "m = GaussianNB().fit(Xtr, ytr)\n" +
                    "theta, var, prior, count = m.theta_, m.var_, m.class_prior_, m.class_count_\n" +
                    "eps = np.array(m.epsilon_)\n" +
                    "jll, logp, p, pred = m.predict_joint_log_proba(Xte), m.predict_log_proba(Xte), m.predict_proba(Xte), m.predict(Xte)");
                NDArray Xtr = py.Xtr, ytr = py.ytr, Xte = py.Xte;
                using var model = new GaussianNaiveBayes().Fit(Xtr, ytr);

                AssertBitIdentical(py.theta, model.Theta, "theta_");
                AssertBitIdentical(py.var, model.Var, "var_");
                AssertBitIdentical(py.prior, model.ClassPrior, "class_prior_");
                AssertBitIdentical(py.count, model.ClassCount, "class_count_");
                AssertBitIdentical(py.eps, np.array(new[] { model.Epsilon }).reshape(new Shape()), "epsilon_");
                AssertBitIdentical(py.jll, model.PredictJointLogProba(Xte), "predict_joint_log_proba");
                AssertBitIdentical(py.logp, model.PredictLogProba(Xte), "predict_log_proba");
                AssertBitIdentical(py.p, model.PredictProba(Xte), "predict_proba");
                AssertEqual(py.pred, model.Predict(Xte), "predict");
            }
        }

        /// <summary>GaussianNB with fixed priors and heavier variance smoothing on the wine dataset (13 features, 3 classes).</summary>
        [TestMethod]
        public void Gaussian_Wine_PriorsAndSmoothing()
        {
            using var scope = NDScope.Open();
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.datasets import load_wine\n" +
                    "from sklearn.naive_bayes import GaussianNB\n" +
                    "X, y = load_wine(return_X_y=True)\n" +
                    "priors = np.array([0.2, 0.5, 0.3])\n" +
                    "m = GaussianNB(priors=priors, var_smoothing=1e-3).fit(X, y)\n" +
                    "var, p = m.var_, m.predict_proba(X)");
                NDArray X = py.X, y = py.y, priors = py.priors;
                using var model = new GaussianNaiveBayes(priors, varSmoothing: 1e-3).Fit(X, y);

                AssertBitIdentical(py.var, model.Var, "var_");
                AssertClose(py.p, model.PredictProba(X), 1e-12, 1e-300, "predict_proba");
            }
        }

        /// <summary>
        ///     BernoulliNB on the digits dataset binarized at 8: the counts and log-probabilities are bit-identical (exact
        ///     integer counts, elementwise logs); the probabilities pass through a matrix product, which scikit-learn
        ///     runs on BLAS, so they agree to rounding.
        /// </summary>
        [TestMethod]
        public void Bernoulli_Digits()
        {
            using var scope = NDScope.Open();
            using (Py.GIL())
            {
                py.Exec(
                    "from sklearn.datasets import load_digits\n" +
                    "from sklearn.naive_bayes import BernoulliNB\n" +
                    "X, y = load_digits(return_X_y=True)\n" +
                    "m = BernoulliNB(alpha=0.5, binarize=8.0).fit(X[:1200], y[:1200])\n" +
                    "fc, flp, clp = m.feature_count_, m.feature_log_prob_, m.class_log_prior_\n" +
                    "Xte = X[1200:].copy()\n" +
                    "logp, pred = m.predict_log_proba(Xte), m.predict(Xte)");
                NDArray X = py.X, y = py.y, Xte = py.Xte;
                using var model = new BernoulliNaiveBayes(alpha: 0.5, binarize: 8.0).Fit(X["0:1200"], y["0:1200"]);

                AssertBitIdentical(py.fc, model.FeatureCount, "feature_count_");
                AssertBitIdentical(py.flp, model.FeatureLogProb, "feature_log_prob_");
                AssertBitIdentical(py.clp, model.ClassLogPrior, "class_log_prior_");
                AssertClose(py.logp, model.PredictLogProba(Xte), 1e-10, 1e-10, "predict_log_proba");
                AssertEqual(py.pred, model.Predict(Xte), "predict");
            }
        }

        /// <summary>
        ///     The Multinomial classifier's conditional log-probabilities, as CherubNLP reads them, equal NumPy's
        ///     evaluation of the same formula <c>log((n_cxv + alpha) / (n_c + n_c * alpha))</c> to the last bit.
        /// </summary>
        [TestMethod]
        public void Multinomial_FormulaMatchesNumPy()
        {
            using var scope = NDScope.Open();
            using (Py.GIL())
            {
                py.Exec(
                    "rng = np.random.default_rng(5)\n" +
                    "X = (rng.random((300, 50)) < 0.15).astype(np.float64)\n" +
                    "y = rng.integers(0, 4, 300)\n" +
                    "alpha = 0.5\n" +
                    "n = np.bincount(y, minlength=4).astype(np.float64)[:, None]\n" +
                    "logp = np.stack([np.log((np.stack([(X[y == c] == v).sum(0) for c in range(4)]) + alpha) / (n + n * alpha)) for v in (0.0, 1.0)], axis=2)");
                NDArray X = py.X, y = py.y, expected = py.logp;
                double[] flat = X.ToArray<double>();
                long[] labels = y.astype(np.int64).ToArray<long>();
                var featureSet = Enumerable.Range(0, 300)
                    .Select(i => Tuple.Create("c" + labels[i], flat.Skip(i * 50).Take(50).ToArray()))
                    .ToList();
                var nb = new MultinomialNaiveBayes(alpha: 0.5) { FeatureSet = featureSet };

                var actual = new double[4 * 50 * 2];
                for (int c = 0; c < 4; c++)
                    for (int x = 0; x < 50; x++)
                        for (int v = 0; v < 2; v++)
                            actual[(c * 50 + x) * 2 + v] = nb.CalCondProb(x, "c" + c, v);
                AssertBitIdentical(expected, np.array(actual).reshape(4, 50, 2), "log P(x = v | c)");
            }
        }
    }
}
