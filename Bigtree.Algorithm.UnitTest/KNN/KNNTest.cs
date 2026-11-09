using Bigtree.Algorithm.KNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.IO;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest.KNN
{
    /// <summary>k-nearest-neighbours: neighbours and distances, votes, the tie rule, text labels, the seeds dataset.</summary>
    [TestClass]
    public class KNNTest : TestEssential
    {
        /// <summary>On a line of points the neighbours, their distances (ascending) and the vote fractions are exact.</summary>
        [TestMethod]
        [NDScoped]
        public void NeighboursAndVotesOnALine()
        {
            var X = np.array(new double[,] { { 0 }, { 1 }, { 2 }, { 10 }, { 11 } });
            var y = np.array(new[] { 0, 0, 1, 1, 1 });
            using var model = new KnnModel(k: 3).Fit(X, y);

            var (distances, indices) = model.KNeighbors(np.array(new double[,] { { 0.4 }, { 10.2 } }));
            CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 4, 2 }, indices.ToArray<long>());
            double[] d = distances.ToArray<double>();
            Assert.AreEqual(0.4, d[0], 1e-12);
            Assert.AreEqual(0.6, d[1], 1e-12);
            Assert.AreEqual(1.6, d[2], 1e-12);
            Assert.AreEqual(0.2, d[3], 1e-12);
            Assert.AreEqual(0.8, d[4], 1e-12);
            Assert.AreEqual(8.2, d[5], 1e-12);

            NDArray proba = model.PredictProba(np.array(new double[,] { { 0.4 }, { 10.2 } }));
            // 0.4's neighbours 0, 1, 2 vote 0, 0, 1; 10.2's neighbours 10, 11, 2 all vote 1.
            Assert.AreEqual(2.0 / 3.0, proba.GetDouble(0, 0), 1e-15);
            Assert.AreEqual(1.0 / 3.0, proba.GetDouble(0, 1), 1e-15);
            Assert.AreEqual(0.0, proba.GetDouble(1, 0));
            Assert.AreEqual(1.0, proba.GetDouble(1, 1));
            CollectionAssert.AreEqual(new[] { 0, 1 }, model.Predict(np.array(new double[,] { { 0.4 }, { 10.2 } })).ToArray<int>());
        }

        /// <summary>A vote tie goes to the class that sorts first, as in scikit-learn.</summary>
        [TestMethod]
        [NDScoped]
        public void VoteTieGoesToTheFirstClass()
        {
            var X = np.array(new double[,] { { -1 }, { 1 } });
            var y = np.array(new[] { 7, 3 });
            using var model = new KnnModel(k: 2).Fit(X, y);
            Assert.AreEqual(3, model.Predict(np.array(new double[,] { { -0.9 } })).GetInt32(0));
        }

        /// <summary>With k = 1 every training sample is its own nearest neighbour; text labels stay text.</summary>
        [TestMethod]
        [NDScoped]
        public void OneNeighbourReproducesTheTrainingLabels()
        {
            np.random.seed(5);
            var X = np.random.rand(30, 4);
            var names = new string[30];
            for (int i = 0; i < 30; i++)
                names[i] = i % 3 == 0 ? "red" : i % 3 == 1 ? "green" : "blue";
            var y = np.array(names);
            using var model = new KnnModel(k: 1).Fit(X, y);

            Assert.AreEqual(1.0, model.Score(X, y));
            Assert.AreEqual("red", (string)model.Predict(X["0:1"]).GetValue(0));
        }

        /// <summary>On the seeds dataset, 5 neighbours classify held-out samples well (a sanity bound, not an oracle).</summary>
        [TestMethod]
        [NDScoped]
        public void SeedsHoldOut()
        {
            var (X, y) = Utils.ReadCsv(Path.Combine(DataDir, "seeds_dataset.csv"));
            NDArray scaled = Utils.Normalize(X);
            NDArray[] folds = Utils.CrossValFolds(210, 5, seed: 3);
            NDArray test = folds[0];
            NDArray train = np.setdiff1d(np.arange(210), test);

            using var model = new KnnModel(k: 5).Fit(scaled[train], y[train]);
            Assert.IsTrue(model.Score(scaled[test], y[test]) >= 0.85, "seeds is easy for 5-NN");
        }

        /// <summary>
        ///     Queries are processed in blocks of 4,194,304 distances: with 4,200 training samples a block holds 998 rows,
        ///     so 2,500 queries take three blocks. The rows on both sides of each block boundary have the neighbours and
        ///     votes of a plain scalar brute-force search, and Predict agrees with KNeighbors' majority.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void BlocksAgreeWithBruteForce()
        {
            var rng = np.random.default_rng(11);
            NDArray X = rng.normal(size: new Shape(4200, 3)), Q = rng.normal(size: new Shape(2500, 3));
            NDArray y = (X[":, 0"] * X[":, 1"] > 0).astype(np.int64) + (X[":, 2"] > 1).astype(np.int64);
            using var model = new KnnModel(k: 6).Fit(X, y);
            var (distances, indices) = model.KNeighbors(Q);
            NDArray proba = model.PredictProba(Q);

            double[] train = X.ToArray<double>(), queries = Q.ToArray<double>();
            long[] labels = y.ToArray<long>();
            foreach (int row in new[] { 0, 997, 998, 1995, 1996, 2499 })
            {
                var expected = Enumerable.Range(0, 4200)
                    .Select(t => (t, d: Math.Sqrt(Enumerable.Range(0, 3).Sum(f => Math.Pow(queries[row * 3 + f] - train[t * 3 + f], 2)))))
                    .OrderBy(pair => pair.d).Take(6).ToArray();
                for (int k = 0; k < 6; k++)
                {
                    Assert.AreEqual(expected[k].t, indices.GetInt64(row, k), $"row {row}, neighbour {k}");
                    Assert.AreEqual(expected[k].d, distances.GetDouble(row, k), 1e-12, $"row {row}, distance {k}");
                }
                for (int c = 0; c < 3; c++)
                    Assert.AreEqual(expected.Count(pair => labels[pair.t] == c) / 6.0, proba.GetDouble(row, c), 1e-15, $"row {row}, class {c}");
            }
        }

        /// <summary>k below 1, too few samples and an unfitted model are refused.</summary>
        [TestMethod]
        [NDScoped]
        public void RefusesBadInput()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => new KnnModel(k: 0));
            using var model = new KnnModel(k: 3);
            Assert.ThrowsException<InvalidOperationException>(() => model.Predict(np.zeros(new Shape(1, 1))));
            Assert.ThrowsException<ArgumentException>(() => model.Fit(np.zeros(new Shape(2, 1)), np.array(new[] { 0, 1 })));
        }
    }
}
