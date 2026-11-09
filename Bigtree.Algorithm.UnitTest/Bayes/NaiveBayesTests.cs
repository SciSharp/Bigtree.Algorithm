using Bigtree.Algorithm.Bayes;
using Bigtree.Algorithm.Statistics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest.Bayes
{
    /// <summary>
    ///     The three Naive Bayes classifiers: the Multinomial one against a verbatim copy of Bigtree 0.4.0's algorithm
    ///     (bit for bit), the Gaussian and Bernoulli ones against hand-computed values and scikit-learn's documented
    ///     examples.
    /// </summary>
    [TestClass]
    public class NaiveBayesTests
    {
        /// <summary>
        ///     Every conditional log-probability CherubNLP asks for (each label × feature × value 0 / 1, plus a value
        ///     that never occurs) equals 0.4.0's to the last bit, and so do the priors and posteriors.
        /// </summary>
        [TestMethod]
        public void Multinomial_MatchesBigtree040BitForBit()
        {
            var random = new Random(20181123);
            var featureSet = Enumerable.Range(0, 120)
                .Select(i => new Tuple<string, double[]>(
                    "label" + random.Next(4),
                    Enumerable.Range(0, 40).Select(f => random.NextDouble() < 0.2 ? 1.0 : 0.0).ToArray()))
                .ToList();
            var labelDist = featureSet.GroupBy(x => x.Item1)
                .Select(g => new Probability { Value = g.Key, Freq = g.Count() })
                .OrderBy(p => p.Value).ToList();

            var current = new MultinomialNaiveBayes { LabelDist = labelDist, FeatureSet = featureSet };
            var legacy = new LegacyMultinomialNaiveBayes { LabelDist = labelDist, FeatureSet = featureSet };

            var condProb = new Dictionary<string, double>();
            foreach (var label in labelDist)
            {
                Assert.AreEqual(Bits(legacy.CalPriorProb(label.Value)), Bits(current.CalPriorProb(label.Value)));
                for (int x = 0; x < 40; x++)
                    foreach (double value in new[] { 0.0, 1.0, 7.5 })
                    {
                        double expected = legacy.CalCondProb(x, label.Value, value);
                        double actual = current.CalCondProb(x, label.Value, value);
                        Assert.AreEqual(Bits(expected), Bits(actual), $"{label.Value} f{x} = {value}: {expected:R} vs {actual:R}");
                        condProb[$"{label.Value} f{x} {value}"] = actual;
                    }
            }

            double[] sample = featureSet[0].Item2;
            foreach (var label in labelDist)
            {
                double prior = current.CalPriorProb(label.Value);
                Assert.AreEqual(Bits(legacy.CalPosteriorProb(label.Value, sample, prior, condProb)),
                                Bits(current.CalPosteriorProb(label.Value, sample, prior, condProb)));
            }
        }

        /// <summary>Features with more than two values count each value exactly, like 0.4.0's equality test.</summary>
        [TestMethod]
        public void Multinomial_MultiValuedFeaturesMatchBigtree040()
        {
            var random = new Random(7);
            var featureSet = Enumerable.Range(0, 60)
                .Select(i => new Tuple<string, double[]>(i % 3 == 0 ? "a" : "b",
                    Enumerable.Range(0, 6).Select(f => (double)random.Next(4)).ToArray()))
                .ToList();
            var current = new MultinomialNaiveBayes(alpha: 1.0) { FeatureSet = featureSet };
            var legacy = new LegacyMultinomialNaiveBayes(alpha: 1.0) { FeatureSet = featureSet };

            foreach (string label in new[] { "a", "b" })
                for (int x = 0; x < 6; x++)
                    for (int value = 0; value < 5; value++)
                        Assert.AreEqual(Bits(legacy.CalCondProb(x, label, value)), Bits(current.CalCondProb(x, label, value)));
        }

        /// <summary>
        ///     The count's equality is 0.4.0's IEEE one, in the SIMD body and the scalar tail alike (37 features: whole
        ///     vectors, then a remainder at any vector width): -0 counts as 0, NaN matches nothing, a NaN request
        ///     counts 0.
        /// </summary>
        [TestMethod]
        public void Multinomial_SignedZeroAndNaNCountLike040()
        {
            double[] palette = { 0.0, -0.0, 1.0, double.NaN, 2.5 };
            var random = new Random(37);
            var featureSet = Enumerable.Range(0, 50)
                .Select(i => new Tuple<string, double[]>("c" + random.Next(3),
                    Enumerable.Range(0, 37).Select(f => palette[random.Next(palette.Length)]).ToArray()))
                .ToList();
            var current = new MultinomialNaiveBayes(alpha: 0.5) { FeatureSet = featureSet };
            var legacy = new LegacyMultinomialNaiveBayes(alpha: 0.5) { FeatureSet = featureSet };

            foreach (string label in featureSet.Select(sample => sample.Item1).Distinct())
                for (int x = 0; x < 37; x++)
                    foreach (double value in palette)
                        Assert.AreEqual(Bits(legacy.CalCondProb(x, label, value)), Bits(current.CalCondProb(x, label, value)),
                            $"{label} f{x} = {value:R}");
        }

        /// <summary>A new FeatureSet list (or one whose Count changed) is counted again.</summary>
        [TestMethod]
        public void Multinomial_RecountsANewFeatureSet()
        {
            var nb = new MultinomialNaiveBayes(alpha: 1.0)
            {
                FeatureSet = new List<Tuple<string, double[]>> { Tuple.Create("a", new[] { 1.0 }), Tuple.Create("a", new[] { 0.0 }) },
            };
            Assert.AreEqual(Math.Log((1 + 1.0) / (2 + 2 * 1.0)), nb.CalCondProb(0, "a", 1.0));

            nb.FeatureSet.Add(Tuple.Create("a", new[] { 1.0 }));          // Count changed: recounted
            Assert.AreEqual(Math.Log((2 + 1.0) / (3 + 3 * 1.0)), nb.CalCondProb(0, "a", 1.0));

            nb.FeatureSet = new List<Tuple<string, double[]>> { Tuple.Create("a", new[] { 0.0 }) };   // another list
            Assert.AreEqual(Math.Log((0 + 1.0) / (1 + 1 * 1.0)), nb.CalCondProb(0, "a", 1.0));
        }

        /// <summary>An unknown label, an out-of-range feature and ragged vectors are refused with a clear exception.</summary>
        [TestMethod]
        public void Multinomial_RefusesBadInput()
        {
            var nb = new MultinomialNaiveBayes
            {
                FeatureSet = new List<Tuple<string, double[]>> { Tuple.Create("a", new[] { 1.0, 0.0 }) },
            };
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => nb.CalCondProb(0, "missing", 1.0));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => nb.CalCondProb(2, "a", 1.0));

            nb.FeatureSet = new List<Tuple<string, double[]>> { Tuple.Create("a", new[] { 1.0 }), Tuple.Create("b", new[] { 1.0, 0.0 }) };
            Assert.ThrowsException<ArgumentException>(() => nb.CalCondProb(0, "a", 1.0));
        }

        /// <summary>
        ///     scikit-learn's GaussianNB docstring example: the per-class means and (smoothed) variances, the priors, and
        ///     the prediction of [-0.8, -1].
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void Gaussian_ScikitLearnDocstringExample()
        {
            var X = np.array(new double[,] { { -1, -1 }, { -2, -1 }, { -3, -2 }, { 1, 1 }, { 2, 1 }, { 3, 2 } });
            var y = np.array(new[] { 1, 1, 1, 2, 2, 2 });
            using var model = new GaussianNaiveBayes().Fit(X, y);

            CollectionAssert.AreEqual(new[] { 1, 2 }, model.Classes.ToArray<int>());
            CollectionAssert.AreEqual(new[] { 3.0, 3.0 }, model.ClassCount.ToArray<double>());
            CollectionAssert.AreEqual(new[] { 0.5, 0.5 }, model.ClassPrior.ToArray<double>());
            // epsilon = 1e-9 * max(var(X, axis=0)) = 1e-9 * var([-1,-2,-3,1,2,3]) = 1e-9 * 14/3
            Assert.AreEqual(1e-9 * 14.0 / 3.0, model.Epsilon, 1e-24);
            AssertClose(new[] { -2.0, -4.0 / 3.0, 2.0, 4.0 / 3.0 }, model.Theta.ToArray<double>(), 1e-15);
            AssertClose(new[] { 2.0 / 3.0, 2.0 / 9.0, 2.0 / 3.0, 2.0 / 9.0 }.Select(v => v + model.Epsilon).ToArray(), model.Var.ToArray<double>(), 1e-15);

            NDArray predicted = model.Predict(np.array(new double[,] { { -0.8, -1 } }));
            Assert.AreEqual(1, predicted.GetInt32(0));
        }

        /// <summary>The probabilities are a softmax of the joint log-likelihood: rows sum to 1 and the logs agree.</summary>
        [TestMethod]
        [NDScoped]
        public void Gaussian_ProbabilitiesAreNormalized()
        {
            np.random.seed(3);
            var X = np.random.randn(40, 3);
            var y = (X[":, 0"] > 0).astype(np.int32) + (X[":, 1"] > 0).astype(np.int32);
            using var model = new GaussianNaiveBayes().Fit(X, y);

            NDArray proba = model.PredictProba(X);
            NDArray log = model.PredictLogProba(X);
            Assert.IsTrue(np.allclose(np.sum(proba, axis: 1), np.ones(new Shape(40))));
            Assert.IsTrue(np.allclose(np.exp(log), proba));
            CollectionAssert.AreEqual(np.argmax(proba, axis: 1).ToArray<long>(),
                                      np.argmax(model.PredictJointLogProba(X), axis: 1).ToArray<long>());
            Assert.IsTrue(model.Score(X, y) > 0.7);
        }

        /// <summary>Text labels stay text, through Fit, Predict and Score.</summary>
        [TestMethod]
        [NDScoped]
        public void Gaussian_TextLabels()
        {
            var X = np.array(new double[,] { { 0.0 }, { 0.2 }, { 5.0 }, { 5.3 } });
            var y = np.array(new[] { "small", "small", "large", "large" });
            using var model = new GaussianNaiveBayes().Fit(X, y);

            CollectionAssert.AreEqual(new[] { "large", "small" }, new[] { (string)model.Classes.GetValue(0), (string)model.Classes.GetValue(1) });
            NDArray predicted = model.Predict(np.array(new double[,] { { 0.1 }, { 4.9 } }));
            Assert.AreEqual("small", (string)predicted.GetValue(0));
            Assert.AreEqual("large", (string)predicted.GetValue(1));
            Assert.AreEqual(1.0, model.Score(X, y));
        }

        /// <summary>Fixed priors are used as given and validated like scikit-learn's.</summary>
        [TestMethod]
        [NDScoped]
        public void Gaussian_FixedPriors()
        {
            var X = np.array(new double[,] { { 0.0 }, { 1.0 }, { 10.0 }, { 11.0 } });
            var y = np.array(new[] { 0, 0, 1, 1 });
            using var model = new GaussianNaiveBayes(priors: np.array(new[] { 0.9, 0.1 })).Fit(X, y);
            CollectionAssert.AreEqual(new[] { 0.9, 0.1 }, model.ClassPrior.ToArray<double>());

            Assert.ThrowsException<ArgumentException>(() => new GaussianNaiveBayes(priors: np.array(new[] { 0.5, 0.6 })).Fit(X, y));
            Assert.ThrowsException<ArgumentException>(() => new GaussianNaiveBayes(priors: np.array(new[] { 1.0 })).Fit(X, y));
        }

        /// <summary>Unfitted use, a wrong feature count, NaN input and mismatched labels are refused.</summary>
        [TestMethod]
        [NDScoped]
        public void Gaussian_RefusesBadInput()
        {
            var X = np.array(new double[,] { { 0.0, 1.0 }, { 1.0, 0.0 } });
            using var model = new GaussianNaiveBayes();
            Assert.ThrowsException<InvalidOperationException>(() => model.Predict(X));
            model.Fit(X, np.array(new[] { 0, 1 }));
            Assert.ThrowsException<ArgumentException>(() => model.Predict(np.array(new double[,] { { 1.0 } })));
            Assert.ThrowsException<ArgumentException>(() => model.Fit(np.array(new double[,] { { double.NaN, 0 } }), np.array(new[] { 0 })));
            Assert.ThrowsException<ArgumentException>(() => model.Fit(X, np.array(new[] { 0, 1, 1 })));
            Assert.ThrowsException<ArgumentException>(() => model.Fit(np.arange(4), np.array(new[] { 0, 1, 1, 0 })));
        }

        /// <summary>
        ///     scikit-learn's BernoulliNB docstring example on NumPy's legacy generator (NumSharp reproduces
        ///     <c>RandomState(1).randint</c>): the third sample is predicted as class 3.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void Bernoulli_ScikitLearnDocstringExample()
        {
            np.random.seed(1);
            var X = np.random.randint(5, size: new Shape(6, 100));
            var Y = np.array(new[] { 1, 2, 3, 4, 4, 5 });
            using var model = new BernoulliNaiveBayes().Fit(X, Y);

            Assert.AreEqual(3, model.Predict(X["2:3"]).GetInt32(0));
            CollectionAssert.AreEqual(new[] { 1, 2, 3, 4, 5 }, model.Classes.ToArray<int>());
        }

        /// <summary>The fitted counts and log-probabilities of a tiny table, by hand.</summary>
        [TestMethod]
        [NDScoped]
        public void Bernoulli_HandComputedProbabilities()
        {
            // Class 0: feature 0 present twice of 2, feature 1 never; class 1: feature 0 once of 1, feature 1 once.
            var X = np.array(new double[,] { { 1, 0 }, { 3, 0 }, { 0.5, 7 } });
            var y = np.array(new[] { 0, 0, 1 });
            using var model = new BernoulliNaiveBayes(alpha: 1.0).Fit(X, y);

            CollectionAssert.AreEqual(new[] { 2.0, 0.0, 1.0, 1.0 }, model.FeatureCount.ToArray<double>());
            AssertClose(new[] { Math.Log(3.0 / 4.0), Math.Log(1.0 / 4.0), Math.Log(2.0 / 3.0), Math.Log(2.0 / 3.0) },
                        model.FeatureLogProb.ToArray<double>(), 1e-15);
            AssertClose(new[] { Math.Log(2.0 / 3.0), Math.Log(1.0 / 3.0) }, model.ClassLogPrior.ToArray<double>(), 1e-15);

            // P(x = [1, 0] | c) by hand, then normalized.
            double joint0 = Math.Log(2.0 / 3.0) + Math.Log(3.0 / 4.0) + Math.Log(1 - 1.0 / 4.0);
            double joint1 = Math.Log(1.0 / 3.0) + Math.Log(2.0 / 3.0) + Math.Log(1 - 2.0 / 3.0);
            NDArray proba = model.PredictProba(np.array(new double[,] { { 2, 0 } }));
            double p0 = Math.Exp(joint0) / (Math.Exp(joint0) + Math.Exp(joint1));
            Assert.AreEqual(p0, proba.GetDouble(0, 0), 1e-12);
        }

        /// <summary>A uniform prior when fitPrior is false, and fixed priors when given.</summary>
        [TestMethod]
        [NDScoped]
        public void Bernoulli_PriorOptions()
        {
            var X = np.array(new double[,] { { 1, 0 }, { 1, 1 }, { 0, 1 } });
            var y = np.array(new[] { 0, 0, 1 });
            using var uniform = new BernoulliNaiveBayes(fitPrior: false).Fit(X, y);
            AssertClose(new[] { Math.Log(0.5), Math.Log(0.5) }, uniform.ClassLogPrior.ToArray<double>(), 1e-15);
            using var fixedPrior = new BernoulliNaiveBayes(classPrior: np.array(new[] { 0.25, 0.75 })).Fit(X, y);
            AssertClose(new[] { Math.Log(0.25), Math.Log(0.75) }, fixedPrior.ClassLogPrior.ToArray<double>(), 1e-15);
        }

        /// <summary>The raw bits of a double, so an assertion fails on a one-ulp difference.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its IEEE 754 bit pattern.</returns>
        private static long Bits(double value) => BitConverter.DoubleToInt64Bits(value);

        /// <summary>Asserts two sequences are equal element by element within an absolute tolerance.</summary>
        /// <param name="expected">The expected values.</param>
        /// <param name="actual">The actual values.</param>
        /// <param name="tolerance">The largest accepted absolute difference.</param>
        private static void AssertClose(double[] expected, double[] actual, double tolerance)
        {
            Assert.AreEqual(expected.Length, actual.Length, "length");
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], actual[i], tolerance, $"element {i}");
        }

        /// <summary>
        ///     Bigtree 0.4.0's <c>MultinomiaNaiveBayes</c>, copied verbatim (commit dfb20dd): the reference the
        ///     vectorized implementation must reproduce bit for bit.
        /// </summary>
        private sealed class LegacyMultinomialNaiveBayes
        {
            /// <summary>The labels and their frequencies.</summary>
            public List<Probability> LabelDist { get; set; }

            /// <summary>The training samples.</summary>
            public List<Tuple<string, double[]>> FeatureSet { get; set; }

            /// <summary>The additive smoothing.</summary>
            private double alpha { get; set; }

            /// <summary>Creates the reference classifier.</summary>
            /// <param name="alpha">The additive smoothing.</param>
            public LegacyMultinomialNaiveBayes(double alpha = 0.5) { this.alpha = alpha; }

            /// <summary>0.4.0's prior.</summary>
            /// <param name="Y">The label.</param>
            /// <returns>The prior.</returns>
            public double CalPriorProb(string Y)
            {
                int N = FeatureSet.Count;
                int k = LabelDist.Count;
                int Nyk = LabelDist.First(x => x.Value == Y).Freq;
                return (Nyk + alpha) / (N + k * alpha);
            }

            /// <summary>0.4.0's conditional log-probability: filter, copy the label's matrix, count.</summary>
            /// <param name="x">The feature.</param>
            /// <param name="Y">The label.</param>
            /// <param name="feature">The value.</param>
            /// <returns>The log-probability.</returns>
            public double CalCondProb(int x, string Y, double feature)
            {
                var featuresIfY = FeatureSet.Where(fd => fd.Item1 == Y).ToList();
                var matrix = ConstructMatrix(featuresIfY);
                int freq = 0;
                for (int y = 0; y < featuresIfY.Count; y++)
                    if (matrix[y, x] == feature)
                        freq++;
                int Nyk = featuresIfY.Count;
                int n = featuresIfY.Count;
                int Nykx = freq;
                return Math.Log((Nykx + alpha) / (Nyk + n * alpha));
            }

            /// <summary>0.4.0's posterior.</summary>
            /// <param name="Y">The label.</param>
            /// <param name="features">The sample.</param>
            /// <param name="priorProb">The prior.</param>
            /// <param name="condProbDictionary">The conditional log-probabilities.</param>
            /// <returns>The posterior.</returns>
            public double CalPosteriorProb(string Y, double[] features, double priorProb, Dictionary<string, double> condProbDictionary)
            {
                int featureCount = features.Length;
                double postProb = Math.Log(priorProb);
                for (int x = 0; x < featureCount; x++)
                {
                    string key = $"{Y} f{x} {features[x]}";
                    if (features[x] == 1)
                        postProb += condProbDictionary[key];
                }

                return Math.Pow(2, postProb);
            }

            /// <summary>0.4.0's per-call matrix copy.</summary>
            /// <param name="featuresIfY">The label's samples.</param>
            /// <returns>The samples as a matrix.</returns>
            private double[,] ConstructMatrix(List<Tuple<string, double[]>> featuresIfY)
            {
                var featureCount = featuresIfY[0].Item2.Length;
                double[,] matrix = new double[featuresIfY.Count, featureCount];
                for (int y = 0; y < featuresIfY.Count; y++)
                    for (int x = 0; x < featureCount; x++)
                        matrix[y, x] = featuresIfY[y].Item2[x];
                return matrix;
            }
        }
    }
}
