using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using Bigtree.Algorithm.NeuralNetwork;
using System.Linq;
using NumSharp;

namespace Bigtree.Algorithm.UnitTest.NeuralNetwork
{
    /// <summary>
    ///     The seeds classification of ankonzoid/NN-scratch, ported: 4-fold cross-validation of a 7-5-3 network on the
    ///     min-max normalized UCI seeds dataset. (In Bigtree 0.4.0 this test could not run: the folds helper returned no
    ///     folds and the test accuracy was commented out.)
    /// </summary>
    [TestClass]
    public class NeuralNetworkClassifierTest : TestEssential
    {
        /// <summary>Trains one network per fold and checks the average training and test accuracy.</summary>
        [TestMethod]
        [NDScoped]
        public void SeedsClassificationTest()
        {
            var filename = Path.Combine(DataDir, "seeds_dataset.csv");
            var n_hidden_nodes = 5; // nodes in hidden layers
            var l_rate = 1.5; // learning rate (per batch of 8: the mean update of 8 samples)
            var n_epochs = 1000; // number of training epochs
            var n_folds = 4; // number of folds for cross-validation
            var batch_size = 8;

            /* =================================
                Read data (X,y) and normalize X
               ================================= */
            Console.WriteLine($"Reading '{filename}'...");
            var (X, y) = Utils.ReadCsv(filename); // read as matrix of floats and int
            // normalize
            X = Utils.Normalize(X);
            // extract shape of X
            int N = (int)X.shape[0], d = (int)X.shape[1];

            var nClasses = (int)np.unique(y).values.size;

            Console.WriteLine($"X.shape = ({N}, {d}), y.shape = ({y.shape[0]},)");
            Console.WriteLine($"size = {X.size}, dimesion = {X.ndim}, number of classes = {nClasses}");
            Console.WriteLine($"hidden nodes = {n_hidden_nodes}, learning rate = {l_rate}, epochs = {n_epochs}");

            /* ===================================
               Create cross-validation folds
               These are a list of a list of indices for each fold
              =================================== */

            var idx_all = np.arange(0, N);
            var idx_folds = Utils.CrossValFolds(N, n_folds, seed: 1);

            /* ===================================
              Train and evaluate the model on each fold
              =================================== */
            Console.WriteLine("Training and cross-validating...");

            List<double> acc_train = new List<double>();
            List<double> acc_test = new List<double>();

            for (int i = 0; i < n_folds; i++)
            {
                // Collect training and test data from folds
                var idx_test = idx_folds[i];
                var idx_train = np.setdiff1d(idx_all, idx_test);
                var X_train = X[idx_train];
                var y_train = y[idx_train];
                var X_test = X[idx_test];
                var y_test = y[idx_test];

                // Build neural network classifier model and train
                using NetworkModel model = new NetworkModel();

                model.Layers.Add(new NeuralLayer(d, "INPUT"));
                model.Layers.Add(new NeuralLayer(n_hidden_nodes, "HIDDEN"));
                model.Layers.Add(new NeuralLayer(nClasses, "OUTPUT"));

                model.Build(seed: i);

                model.Train(X_train, y_train, iterations: n_epochs, learningRate: l_rate, batchSize: batch_size);

                // Make predictions for training and test data
                var y_train_predict = model.Predict(X_train);
                var y_test_predict = model.Predict(X_test);

                acc_train.Add(100 * np.mean((y_train == y_train_predict).astype(np.float64)).GetDouble());
                acc_test.Add(100 * np.mean((y_test == y_test_predict).astype(np.float64)).GetDouble());

                Console.WriteLine($"Fold {i + 1}/{n_folds}: train acc = {acc_train.Last():F2}%, test acc = {acc_test.Last():F2}% (n_train = {X_train.shape[0]}, n_test = {X_test.shape[0]})");
            }

            Console.WriteLine($"Avg train acc = {acc_train.Average():F2}%, avg test acc = {acc_test.Average():F2}%");
            Assert.IsTrue(acc_train.Average() > 90, $"average training accuracy {acc_train.Average():F2}%");
            Assert.IsTrue(acc_test.Average() > 85, $"average test accuracy {acc_test.Average():F2}%");
        }
    }
}
