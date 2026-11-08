using System;
using System.Linq;
using Bigtree.Algorithm.SVM;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Bigtree.Algorithm.UnitTest.SVM
{
    /// <summary>
    ///     The SVM's NumSharp entry points: <see cref="Problem.FromNDArray"/> and the batch predictions, which must agree
    ///     with the per-row libsvm path for every kernel.
    /// </summary>
    [TestClass]
    public class NDArrayTests
    {
        /// <summary>A dense matrix becomes 1-based nodes, zeros included, with MaxIndex = the feature count.</summary>
        [TestMethod]
        public void FromNDArrayIsDenseAndOneBased()
        {
            using var scope = NDScope.Open();
            Problem problem = Problem.FromNDArray(np.array(new double[,] { { 0.5, 0 }, { -1, 2 } }), np.array(new[] { 1, -1 }));

            Assert.AreEqual(2, problem.Count);
            Assert.AreEqual(2, problem.MaxIndex);
            CollectionAssert.AreEqual(new[] { 1.0, -1.0 }, problem.Y);
            Assert.AreEqual(1, problem.X[0][0].Index);
            Assert.AreEqual(2, problem.X[0][1].Index);
            Assert.AreEqual(0.0, problem.X[0][1].Value);
            Assert.AreEqual(2.0, problem.X[1][1].Value);
        }

        /// <summary>
        ///     For every SVM type and kernel, the batch labels equal the per-row ones and the batch values match the
        ///     per-row regression values. Before the polynomial-kernel fix, POLY's per-row predictions used
        ///     <c>Degree</c> where training used <c>Gamma</c>, and this test failed for it.
        /// </summary>
        [TestMethod]
        public void BatchPredictionsAgreeWithPerRow()
        {
            using var scope = NDScope.Open();
            np.random.seed(17);
            NDArray X = np.random.uniform(-1.0, 1.0, new Shape(120, 3));
            NDArray labels = (X[":, 0"] + X[":, 1"] * X[":, 2"] > 0).astype(np.float64) + (X[":, 2"] > 0.5).astype(np.float64);
            NDArray values = np.sin(3 * X[":, 0"]) + X[":, 1"];
            NDArray query = np.random.uniform(-1.2, 1.2, new Shape(40, 3));

            foreach (KernelType kernel in new[] { KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID })
            {
                foreach (SvmType type in new[] { SvmType.C_SVC, SvmType.NU_SVC, SvmType.EPSILON_SVR, SvmType.NU_SVR })
                {
                    bool regression = type == SvmType.EPSILON_SVR || type == SvmType.NU_SVR;
                    Problem train = Problem.FromNDArray(X, regression ? values : labels);
                    var parameter = new Parameter { SvmType = type, KernelType = kernel, Gamma = 0.7, Degree = 3, Coefficient0 = 0.5, Nu = 0.3 };
                    Model model = Training.Train(train, parameter);

                    double[] perRow = Problem.FromNDArray(query, np.zeros(new Shape(40))).X.Select(model.Predict).ToArray();
                    double[] batch = model.PredictLabels(query).ToArray<double>();
                    for (int i = 0; i < perRow.Length; i++)
                    {
                        if (regression)
                            Assert.AreEqual(perRow[i], batch[i], 1e-9 * (1 + Math.Abs(perRow[i])), $"{type} {kernel} row {i}");
                        else
                            Assert.AreEqual(perRow[i], batch[i], $"{type} {kernel} row {i}");
                    }
                }
            }
        }

        /// <summary>Two classes give one decision column; its sign decides the label like libsvm's vote.</summary>
        [TestMethod]
        public void TwoClassDecisionValues()
        {
            using var scope = NDScope.Open();
            var X = np.array(new double[,] { { -2, 0 }, { -1, 1 }, { 1, -1 }, { 2, 0 } });
            Model model = Training.Train(Problem.FromNDArray(X, np.array(new[] { 4, 4, 9, 9 })), new Parameter { KernelType = KernelType.LINEAR });

            NDArray decision = model.DecisionValues(X);
            CollectionAssert.AreEqual(new long[] { 4, 1 }, decision.shape);
            NDArray predicted = model.PredictLabels(X);
            double first = model.ClassLabels[0], second = model.ClassLabels[1];
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(decision.GetDouble(i, 0) > 0 ? first : second, predicted.GetDouble(i));
            CollectionAssert.AreEqual(new[] { 4.0, 4, 9, 9 }, predicted.ToArray<double>());
        }

        /// <summary>The batch probabilities equal the per-row ones and sum to 1.</summary>
        [TestMethod]
        public void BatchProbabilitiesAgreeWithPerRow()
        {
            using var scope = NDScope.Open();
            np.random.seed(5);
            NDArray X = np.random.randn(60, 2);
            NDArray y = (X[":, 0"] > 0).astype(np.float64);
            Training.SetRandomSeed(1);
            Model model = Training.Train(Problem.FromNDArray(X, y), new Parameter { Probability = true });

            NDArray batch = model.PredictLabelsProbability(X["0:5"]);
            double[][] perRow = model.PredictLabelsProbability(Problem.FromNDArray(X["0:5"], np.zeros(new Shape(5))));
            for (int i = 0; i < 5; i++)
                for (int c = 0; c < 2; c++)
                    Assert.AreEqual(perRow[i][c], batch.GetDouble(i, c));
            Assert.IsTrue(np.allclose(np.sum(batch, axis: 1), np.ones(new Shape(5))));
        }

        /// <summary>A model trained on 0-based features (CherubNLP's) cannot read a matrix: index 0 has no column.</summary>
        [TestMethod]
        public void ZeroBasedModelsUseTheNodeApi()
        {
            using var scope = NDScope.Open();
            var problem = new Problem(4, new[] { 0.0, 0, 1, 1 },
                new[] { new[] { new Node(0, -1.0) }, new[] { new Node(0, -2.0) }, new[] { new Node(0, 1.0) }, new[] { new Node(0, 2.0) } }, 1);
            Model model = Training.Train(problem, new Parameter { KernelType = KernelType.LINEAR });

            Assert.AreEqual(1.0, model.Predict(new[] { new Node(0, 3.0) }));
            Assert.ThrowsException<InvalidOperationException>(() => model.PredictLabels(np.array(new double[,] { { 3.0 } })));
        }
    }
}
