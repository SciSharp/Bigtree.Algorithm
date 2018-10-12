using Bigtree.Algorithm.NeuralNetwork;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.NeuralNetwork
{
    [TestClass]
    public class NetworkModelTest
    {
        /// <summary>
        /// AND gate test
        /// </summary>
        [TestMethod]
        public void TestAndGate()
        {
            NetworkModel model = new NetworkModel();
            model.Layers.Add(new NeuralLayer(2, 0.1, "INPUT"));
            model.Layers.Add(new NeuralLayer(2, 0.1, "HIDDEN"));
            model.Layers.Add(new NeuralLayer(1, 0.1, "OUTPUT"));

            model.Build();

            NeuralData X = new NeuralData(4);
            X.Add(0, 0);
            X.Add(0, 1);
            X.Add(1, 0);
            X.Add(1, 1);

            NeuralData Y = new NeuralData(4);
            Y.Add(0);
            Y.Add(0);
            Y.Add(0);
            Y.Add(1);

            // model.Train(X, Y, iterations: 10, learningRate: 0.1);
        }
    }
}
