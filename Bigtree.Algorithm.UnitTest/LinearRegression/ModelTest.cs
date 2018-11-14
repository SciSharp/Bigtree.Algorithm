using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.LinearRegression
{
    [TestClass]
    public class ModelTest
    {
        [TestMethod]
        public void Regression()
        {
            var np = new NumPy<double>();
            var data_x = np.linspace(1, 10, 100);
            var data_y = np.sin(data_x);// + 0.1 * np.power;
        }
    }
}
