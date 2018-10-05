using Bigtree.Algorithm.Maths;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.Maths
{
    [TestClass]
    public class SigmoidTest
    {
        [TestMethod]
        public void CalTest()
        {
            var x = new Random().NextDouble();
            var sigmoid = new Sigmoid(x);
            Assert.IsTrue(0 < sigmoid.y && sigmoid.y < 1);
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
            Campy.Parallel.For(n, i => y[i] = 1 / (1 + Math.Pow(Math.E, -x[i])));
#endif
            for (int i = 0; i < n; ++i)
            {
                Assert.IsTrue(0 < y[i] && y[i] < 1);
            }
        }
    }
}
