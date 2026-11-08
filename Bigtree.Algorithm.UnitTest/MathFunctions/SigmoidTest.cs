using Bigtree.Algorithm.MathFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.MathFunctions
{
    [TestClass]
    public class SigmoidTest
    {
        [TestMethod]
        public void CalTest()
        {
            var x = new Random().NextDouble();
            var y = Function.Sigmoid(x);
            Assert.IsTrue(0 < y && y < 1);
        }

        /// <summary>The NDArray sigmoid and its derivative agree with the scalar functions element by element.</summary>
        [TestMethod]
        public void ArraySigmoidMatchesScalar()
        {
            using var scope = NumSharp.NDScope.Open();
            double[] inputs = { -30, -2.5, -0.1, 0, 0.1, 2.5, 30 };
            NumSharp.NDArray y = Function.Sigmoid(NumSharp.np.array(inputs), coefficient: 1.5);
            NumSharp.NDArray dy = Function.SigmoidDerivative(y);
            for (int i = 0; i < inputs.Length; i++)
            {
                Assert.AreEqual(Function.Sigmoid(inputs[i], 1.5), y.GetDouble(i), 1e-15);
                Assert.AreEqual(Function.SigmoidDerivative(Function.Sigmoid(inputs[i], 1.5)), dy.GetDouble(i), 1e-15);
            }
        }

        public void CudaTest()
        {
            int n = 10;
            var rand = new Random();
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = rand.NextDouble();
            }
#if CUDA
            Campy.Parallel.For(n, i => y[i] = Function.Sigmoid(-x[i]));
#endif
            for (int i = 0; i < n; ++i)
            {
                Assert.IsTrue(0 < y[i] && y[i] < 1);
            }
        }
    }
}
