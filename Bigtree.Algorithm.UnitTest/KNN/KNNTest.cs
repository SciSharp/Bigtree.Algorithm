using Microsoft.VisualStudio.TestTools.UnitTesting;
using PandasNET;
using System;
using System.Collections.Generic;
using System.Text;
using PandasNET.Extensions;

namespace Bigtree.Algorithm.UnitTest.KNN
{
    [TestClass]
    public class KNNTest
    {
        [TestMethod]
        public void TestKNN()
        {
            string filePath = "./data/train.csv";

            var pd = new Pandas();
            var dataFrame = pd.read_csv(filePath);
        }
    }
}
