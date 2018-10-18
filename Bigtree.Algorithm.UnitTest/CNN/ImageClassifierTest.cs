using Microsoft.VisualStudio.TestTools.UnitTesting;
using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Extensions;

namespace Bigtree.Algorithm.UnitTest.CNN
{
    [TestClass]
    public class ImageClassifierTest
    {
        [TestMethod]
        public void MNIST()
        {
            (var X, var y_dash) = new FileReaderMNIST().LoadImagesAndLables("data/t10k-labels-idx1-ubyte.gz", "data/t10k-images-idx3-ubyte.gz");
            X = X.Minus((int)X.Mean()[0]);
        }
    }
}
