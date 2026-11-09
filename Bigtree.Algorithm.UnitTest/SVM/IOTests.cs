using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Text;
using Bigtree.Algorithm.SVM;
using Bigtree.Algorithm.SVMTest;

namespace Bigtree.Algorithm.UnitTest.SVM
{
    /// <summary>
    ///     Reading and writing problems, models and range transforms. Bigtree 0.4.0 compared against golden files
    ///     (<c>train0.problem</c>, <c>svm0.model</c>) that were never committed; these tests write, read back and compare
    ///     instead, plus pin the text format of a small problem.
    /// </summary>
    [TestClass]
    public class IOTests
    {
        /// <summary>A problem written and read back is equal to the original (values to the format's 6 decimals).</summary>
        [TestMethod]
        public void ProblemRoundTrip()
        {
            Problem expected = Rounded(SVMUtilities.CreateTwoClassProblem(100));
            using (MemoryStream stream = new MemoryStream())
            {
                Problem.Write(stream, expected);
                stream.Position = 0;
                Problem actual = Problem.Read(stream);
                Assert.AreEqual(expected, actual);
            }
        }

        /// <summary>The libsvm text format: label, then index:value pairs with six decimals.</summary>
        [TestMethod]
        public void WriteProblem()
        {
            var problem = new Problem(2, new[] { 1.0, -1.0 },
                new[] { new[] { new Node(1, 0.5), new Node(2, -2) }, new[] { new Node(1, 3.25) } }, 2);
            using (MemoryStream stream = new MemoryStream())
            {
                Problem.Write(stream, problem);
                Assert.AreEqual("1 1:0.500000 2:-2.000000\n-1 1:3.250000\n", Encoding.ASCII.GetString(stream.ToArray()));
            }
        }

        /// <summary>
        ///     A trained model written and read back is the same model to the last bit (0.4.0 wrote six decimals, which
        ///     changed rho and the coefficients).
        /// </summary>
        [TestMethod]
        public void ModelRoundTrip()
        {
            Problem train = SVMUtilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.KernelType = KernelType.LINEAR;

            Training.SetRandomSeed(SVMUtilities.TRAINING_SEED);
            Model expected = Training.Train(scaled, param);
            using (MemoryStream stream = new MemoryStream())
            {
                Model.Write(stream, expected);
                stream.Position = 0;
                Model actual = Model.Read(stream);
                Assert.AreEqual(expected.NumberOfClasses, actual.NumberOfClasses);
                Assert.AreEqual(expected.SupportVectorCount, actual.SupportVectorCount);
                CollectionAssert.AreEqual(expected.ClassLabels, actual.ClassLabels);
                CollectionAssert.AreEqual(expected.NumberOfSVPerClass, actual.NumberOfSVPerClass);
                Assert.AreEqual(expected.Rho[0], actual.Rho[0]);
                for (int i = 0; i < expected.SupportVectorCount; i++)
                {
                    Assert.AreEqual(expected.SupportVectorCoefficients[0][i], actual.SupportVectorCoefficients[0][i]);
                    for (int j = 0; j < expected.SupportVectors[i].Length; j++)
                        Assert.AreEqual(expected.SupportVectors[i][j].Value, actual.SupportVectors[i][j].Value);
                }
                foreach (Node[] sample in scaled.X)
                    Assert.AreEqual(expected.Predict(sample), actual.Predict(sample));
            }
        }

        /// <summary>
        ///     A range transform of a 1-based problem (libsvm's convention) scales every feature into [-1, 1]. In
        ///     0.4.0 the transform was sized for 0-based indices and the last feature crashed <c>Transform</c>.
        /// </summary>
        [TestMethod]
        public void RangeTransformScalesOneBasedProblems()
        {
            Problem train = SVMUtilities.CreateTwoClassProblem(100);
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            foreach (Node[] sample in scaled.X)
                foreach (Node node in sample)
                    Assert.IsTrue(node.Value >= -1 - 1e-12 && node.Value <= 1 + 1e-12, $"feature {node.Index} = {node.Value}");

            using (MemoryStream stream = new MemoryStream())
            {
                RangeTransform.Write(stream, transform);
                stream.Position = 0;
                RangeTransform read = RangeTransform.Read(stream);
                Assert.AreEqual(transform.Transform(12.5, 2), read.Transform(12.5, 2), 1e-12);
            }
        }

        /// <summary>
        ///     A 0-based problem (CherubNLP's convention: index = column, MaxIndex = feature count) keeps the transform
        ///     length it had in 0.4.0, so transforms saved by 0.4.0 still read and apply.
        /// </summary>
        [TestMethod]
        public void RangeTransformKeepsZeroBasedLength()
        {
            var problem = new Problem(2, new[] { 0.0, 1.0 },
                new[] { new[] { new Node(0, 1.0), new Node(1, 5.0) }, new[] { new Node(0, 3.0), new Node(1, 7.0) } }, 2);
            RangeTransform transform = RangeTransform.Compute(problem);
            using (MemoryStream stream = new MemoryStream())
            {
                RangeTransform.Write(stream, transform);
                string text = Encoding.ASCII.GetString(stream.ToArray());
                Assert.IsTrue(text.StartsWith("2\n") || text.StartsWith("2\r\n"), text);
            }
            Assert.AreEqual(-1.0, transform.Transform(1.0, 0));
            Assert.AreEqual(1.0, transform.Transform(7.0, 1));
        }

        /// <summary>The problem with every value rounded to the six decimals the text format keeps.</summary>
        /// <param name="problem">The problem.</param>
        /// <returns>A copy with rounded values.</returns>
        private static Problem Rounded(Problem problem)
        {
            var x = new Node[problem.Count][];
            for (int i = 0; i < problem.Count; i++)
            {
                x[i] = new Node[problem.X[i].Length];
                for (int j = 0; j < x[i].Length; j++)
                    x[i][j] = new Node(problem.X[i][j].Index, Math.Round(problem.X[i][j].Value, 6));
            }
            return new Problem(problem.Count, (double[])problem.Y.Clone(), x, problem.MaxIndex);
        }
    }
}
