using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.IO;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest
{
    /// <summary>The dataset helpers: CSV reading, cross-validation folds and min-max normalization.</summary>
    [TestClass]
    public class UtilsTests : TestEssential
    {
        /// <summary>The seeds dataset reads as 210 samples of 7 features and three balanced classes numbered 0..2.</summary>
        [TestMethod]
        [NDScoped]
        public void ReadCsv_Seeds()
        {
            var (X, y) = Utils.ReadCsv(Path.Combine(DataDir, "seeds_dataset.csv"));

            CollectionAssert.AreEqual(new long[] { 210, 7 }, X.shape);
            Assert.AreEqual(NPTypeCode.Double, X.typecode);
            Assert.AreEqual(NPTypeCode.Int32, y.typecode);
            CollectionAssert.AreEqual(new long[] { 70, 70, 70 }, np.bincount(y).ToArray<long>());
            Assert.AreEqual(15.26, X.GetDouble(0, 0));
            Assert.AreEqual(5.22, X.GetDouble(0, 6));
        }

        /// <summary>The folds are disjoint, equally sized (the remainder is left out), in range and reproducible.</summary>
        [TestMethod]
        [NDScoped]
        public void CrossValFolds_DisjointAndReproducible()
        {
            NDArray[] folds = Utils.CrossValFolds(210, 4, seed: 7);

            Assert.AreEqual(4, folds.Length);
            Assert.IsTrue(folds.All(fold => fold.size == 52));
            long[] all = folds.SelectMany(fold => fold.ToArray<long>()).ToArray();
            Assert.AreEqual(208, all.Distinct().Count(), "no index is in two folds");
            Assert.IsTrue(all.All(index => index >= 0 && index < 210));

            NDArray[] again = Utils.CrossValFolds(210, 4, seed: 7);
            for (int i = 0; i < 4; i++)
                CollectionAssert.AreEqual(folds[i].ToArray<long>(), again[i].ToArray<long>());

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => Utils.CrossValFolds(3, 4));
        }

        /// <summary>Every column of a normalized matrix spans exactly [0, 1].</summary>
        [TestMethod]
        [NDScoped]
        public void Normalize_ScalesEveryColumnToUnitRange()
        {
            var X = np.array(new double[,] { { 1, 10 }, { 3, -10 }, { 2, 0 } });
            NDArray scaled = Utils.Normalize(X);

            CollectionAssert.AreEqual(new[] { 0.0, 1.0, 1.0, 0.0, 0.5, 0.5 }, scaled.ToArray<double>());
            CollectionAssert.AreEqual(new[] { 1.0, 10, 3, -10, 2, 0 }, X.ToArray<double>(), "the input is left alone");
        }
    }
}
