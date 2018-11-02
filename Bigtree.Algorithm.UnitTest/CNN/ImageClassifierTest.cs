using Microsoft.VisualStudio.TestTools.UnitTesting;
using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Extensions;
using NumSharp;
using Bigtree.Algorithm.CNN;

namespace Bigtree.Algorithm.UnitTest.CNN
{
    [TestClass]
    public class ImageClassifierTest
    {
        [TestMethod]
        public void MNIST()
        {
            /*(var X, var y_dash) = new FileReaderMNIST().LoadImagesAndLables("data/t10k-labels-idx1-ubyte.gz", "data/t10k-images-idx3-ubyte.gz", 100);
            X = X.Minus((int)X.Mean()[0]);
            X = X.Divide((int)X.Std()[0]);
            var test_data = X.HStack(y_dash);

            (X, y_dash) = new FileReaderMNIST().LoadImagesAndLables("data/train-labels-idx1-ubyte.gz", "data/train-images-idx3-ubyte.gz", 500);
            X = X.Minus((int)X.Mean()[0]);
            X = X.Divide((int)X.Std()[0]);
            var train_data = X.HStack(y_dash);

            new NDArrayRandom().Shuffle(train_data);

            var cnn = new NetworkModel();
            cnn.Train();*/
        }
    }
}
