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
