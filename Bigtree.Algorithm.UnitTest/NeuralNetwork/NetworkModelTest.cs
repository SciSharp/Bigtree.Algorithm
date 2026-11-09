using Bigtree.Algorithm.NeuralNetwork;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;

namespace Bigtree.Algorithm.UnitTest.NeuralNetwork
{
    /// <summary>
    ///     The matrix network: it learns the AND gate, batch size 1 is exactly per-sample backpropagation (checked
    ///     against a plain scalar implementation), and bad input is refused.
    /// </summary>
    [TestClass]
    public class NetworkModelTest
    {
        /// <summary>
        /// AND gate test. The network has no biases, so a constant third input (always 1) plays the bias; the two
        /// output neurons are the classes false and true.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void TestAndGate()
        {
            using var model = new NetworkModel();
            model.Layers.Add(new NeuralLayer(3, "INPUT"));
            model.Layers.Add(new NeuralLayer(4, "HIDDEN"));
            model.Layers.Add(new NeuralLayer(2, "OUTPUT"));
            model.Build(seed: 1);

            var X = np.array(new double[,] { { 0, 0, 1 }, { 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 } });
            var Y = np.array(new[] { 0, 0, 0, 1 });
            model.Train(X, Y, iterations: 3000, learningRate: 2.0, batchSize: 4);

            CollectionAssert.AreEqual(new[] { 0, 0, 0, 1 }, model.Predict(X).ToArray<int>());
        }

        /// <summary>
        ///     batchSize 1 reproduces textbook per-sample backpropagation (NN-scratch's), computed here with plain
        ///     loops from the same initial weights: every weight agrees after 50 epochs.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void BatchSizeOneIsPerSampleBackpropagation()
        {
            np.random.seed(4);
            var X = np.random.rand(12, 3);
            var Y = np.array(new[] { 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2 });
            using var model = new NetworkModel();
            model.Layers.Add(new NeuralLayer(3));
            model.Layers.Add(new NeuralLayer(4));
            model.Layers.Add(new NeuralLayer(3));
            model.Build(seed: 9);
            double[][] weights = { model.Weights[0].ToArray<double>(), model.Weights[1].ToArray<double>() };

            model.Train(X, Y, iterations: 50, learningRate: 0.5, batchSize: 1);
            ScalarBackpropagation(X.ToArray<double>(), Y.ToArray<int>(), weights, new[] { 3, 4, 3 }, epochs: 50, rate: 0.5);

            for (int l = 0; l < 2; l++)
            {
                double[] actual = model.Weights[l].ToArray<double>();
                for (int i = 0; i < actual.Length; i++)
                    Assert.AreEqual(weights[l][i], actual[i], 1e-12, $"layer {l} weight {i}");
            }
        }

        /// <summary>A mini-batch update is the mean of its samples' updates at the batch's starting weights.</summary>
        [TestMethod]
        [NDScoped]
        public void MiniBatchUpdateIsTheMeanOfTheSampleUpdates()
        {
            var X = np.array(new double[,] { { 0.2, 0.9 }, { 0.7, 0.1 } });
            var Y = np.array(new[] { 1, 0 });

            using var batched = Network(seed: 3);
            using var first = Network(seed: 3);
            using var second = Network(seed: 3);
            NDArray start = batched.Weights[0].copy();
            batched.Train(X, Y, iterations: 1, learningRate: 1.0, batchSize: 2);
            first.Train(X["0:1"], Y["0:1"], iterations: 1, learningRate: 1.0, batchSize: 1);
            second.Train(X["1:2"], Y["1:2"], iterations: 1, learningRate: 1.0, batchSize: 1);

            NDArray mean = start + ((first.Weights[0] - start) + (second.Weights[0] - start)) / 2.0;
            Assert.IsTrue(np.allclose(batched.Weights[0], mean, rtol: 0, atol: 1e-15));
        }

        /// <summary>Training before Build, a wrong input width and out-of-range classes are refused.</summary>
        [TestMethod]
        [NDScoped]
        public void RefusesBadInput()
        {
            using var model = new NetworkModel();
            model.AddLayer(new NeuralLayer(2));
            model.AddLayer(new NeuralLayer(2));
            var X = np.array(new double[,] { { 0, 1 } });
            Assert.ThrowsException<InvalidOperationException>(() => model.Train(X, np.array(new[] { 0 }), 1));
            model.Build(seed: 1);
            Assert.ThrowsException<ArgumentException>(() => model.Train(np.array(new double[,] { { 0, 1, 2 } }), np.array(new[] { 0 }), 1));
            Assert.ThrowsException<ArgumentException>(() => model.Train(X, np.array(new[] { 2 }), 1));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => model.Train(X, np.array(new[] { 0 }), 1, batchSize: 0));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => new NeuralLayer(0));
        }

        /// <summary>A built 2-3-2 network with the given seed.</summary>
        /// <param name="seed">The weight seed.</param>
        /// <returns>The network; the caller disposes it.</returns>
        private static NetworkModel Network(int seed)
        {
            var model = new NetworkModel();
            model.AddLayer(new NeuralLayer(2));
            model.AddLayer(new NeuralLayer(3));
            model.AddLayer(new NeuralLayer(2));
            model.Build(seed);
            return model;
        }

        /// <summary>
        ///     Per-sample backpropagation with plain loops (NN-scratch's algorithm): forward, deltas from the output
        ///     back with the pre-update weights, then each layer's update from its own input.
        /// </summary>
        /// <param name="X">The samples, row-major (samples × sizes[0]).</param>
        /// <param name="Y">The class of every sample.</param>
        /// <param name="weights">The row-major weight matrices, updated in place.</param>
        /// <param name="sizes">The neurons per layer.</param>
        /// <param name="epochs">The passes over the samples.</param>
        /// <param name="rate">The learning rate.</param>
        private static void ScalarBackpropagation(double[] X, int[] Y, double[][] weights, int[] sizes, int epochs, double rate)
        {
            int layers = sizes.Length;
            for (int epoch = 0; epoch < epochs; epoch++)
                for (int s = 0; s < Y.Length; s++)
                {
                    var outputs = new double[layers][];
                    outputs[0] = new double[sizes[0]];
                    Array.Copy(X, s * sizes[0], outputs[0], 0, sizes[0]);
                    for (int l = 1; l < layers; l++)
                    {
                        outputs[l] = new double[sizes[l]];
                        for (int j = 0; j < sizes[l]; j++)
                        {
                            double sum = 0;
                            for (int i = 0; i < sizes[l - 1]; i++)
                                sum += weights[l - 1][j * sizes[l - 1] + i] * outputs[l - 1][i];
                            outputs[l][j] = 1 / (1 + Math.Exp(-sum));
                        }
                    }

                    var deltas = new double[layers][];
                    deltas[layers - 1] = new double[sizes[layers - 1]];
                    for (int j = 0; j < sizes[layers - 1]; j++)
                    {
                        double o = outputs[layers - 1][j];
                        deltas[layers - 1][j] = ((Y[s] == j ? 1 : 0) - o) * o * (1 - o);
                    }
                    for (int l = layers - 2; l >= 1; l--)
                    {
                        deltas[l] = new double[sizes[l]];
                        for (int j = 0; j < sizes[l]; j++)
                        {
                            double error = 0;
                            for (int k = 0; k < sizes[l + 1]; k++)
                                error += weights[l][k * sizes[l] + j] * deltas[l + 1][k];
                            deltas[l][j] = error * outputs[l][j] * (1 - outputs[l][j]);
                        }
                    }
                    for (int l = 1; l < layers; l++)
                        for (int j = 0; j < sizes[l]; j++)
                            for (int i = 0; i < sizes[l - 1]; i++)
                                weights[l - 1][j * sizes[l - 1] + i] += rate * deltas[l][j] * outputs[l - 1][i];
                }
        }
    }
}
