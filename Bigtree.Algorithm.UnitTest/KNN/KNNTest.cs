using Bigtree.Algorithm.KNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.IO;

namespace Bigtree.Algorithm.UnitTest.KNN
{
    /// <summary>k-nearest-neighbours: neighbours and distances, votes, the tie rule, text labels, the seeds dataset.</summary>
    [TestClass]
    public class KNNTest : TestEssential
    {
        /// <summary>On a line of points the neighbours, their distances (ascending) and the vote fractions are exact.</summary>
        [TestMethod]
        public void NeighboursAndVotesOnALine()
        {
            using var scope = NDScope.Open();
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
        public void VoteTieGoesToTheFirstClass()
        {
            using var scope = NDScope.Open();
            var X = np.array(new double[,] { { -1 }, { 1 } });
            var y = np.array(new[] { 7, 3 });
            using var model = new KnnModel(k: 2).Fit(X, y);
            Assert.AreEqual(3, model.Predict(np.array(new double[,] { { -0.9 } })).GetInt32(0));
        }

        /// <summary>With k = 1 every training sample is its own nearest neighbour; text labels stay text.</summary>
        [TestMethod]
        public void OneNeighbourReproducesTheTrainingLabels()
        {
            using var scope = NDScope.Open();
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
        public void SeedsHoldOut()
        {
            using var scope = NDScope.Open();
            var (X, y) = Utils.ReadCsv(Path.Combine(DataDir, "seeds_dataset.csv"));
            NDArray scaled = Utils.Normalize(X);
            NDArray[] folds = Utils.CrossValFolds(210, 5, seed: 3);
            NDArray test = folds[0];
            NDArray train = np.setdiff1d(np.arange(210), test);

            using var model = new KnnModel(k: 5).Fit(scaled[train], y[train]);
            Assert.IsTrue(model.Score(scaled[test], y[test]) >= 0.85, "seeds is easy for 5-NN");
        }

        /// <summary>k below 1, too few samples and an unfitted model are refused.</summary>
        [TestMethod]
        public void RefusesBadInput()
        {
            using var scope = NDScope.Open();
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => new KnnModel(k: 0));
            using var model = new KnnModel(k: 3);
            Assert.ThrowsException<InvalidOperationException>(() => model.Predict(np.zeros(new Shape(1, 1))));
            Assert.ThrowsException<ArgumentException>(() => model.Fit(np.zeros(new Shape(2, 1)), np.array(new[] { 0, 1 })));
        }
    }
}
