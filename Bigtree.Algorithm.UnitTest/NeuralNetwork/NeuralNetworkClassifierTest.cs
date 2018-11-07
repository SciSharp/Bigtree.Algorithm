using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NumSharp.Extensions;
using NumSharp;
using Bigtree.Algorithm.NeuralNetwork;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest.NeuralNetwork
{
    [TestClass]
    public class NeuralNetworkClassifierTest
    {
        [TestMethod]
        public void SeedsClassificationTest()
        {
            var filename = @"data/seeds_dataset.csv";
            var np = new NumPy<int>();
            var n_hidden_nodes = 5; // nodes in hidden layers
            var l_rate = 0.6; // learning rate
            var n_epochs = 1000; // number of training epochs
            var n_folds = 4; // number of folds for cross-validation

            /* =================================
                Read data (X,y) and normalize X
               ================================= */
            Console.WriteLine($"Reading '{filename}'...");
            (var X, var y) = Utils.ReadCsv(filename); // read as matrix of floats and int
            // normalize
            X.Normalize();
            // extract shape of X
            (var N, var d) = X.Shape.BiShape;

            var nClasses = y.Unique().Size;

            Console.WriteLine($"X.shape = {X.Shape}, y.shape = {y.Shape}");
            Console.WriteLine($"size = {X.Size}, dimesion = {X.NDim}, number of classes = {nClasses}");
            Console.WriteLine($"hidden nodes = {n_hidden_nodes}, learning rate = {l_rate}, epochs = {n_epochs}");

            /* ===================================
               Create cross-validation folds
               These are a list of a list of indices for each fold
              =================================== */

            var idx_all = np.arange(0, N);
            var idx_folds = Utils.CrossValFolds(N, 4);

            /* ===================================
              Train and evaluate the model on each fold
              =================================== */
            // List<acc_train, acc_test = list(), list()  # training/test accuracy score
            Console.WriteLine("Training and cross-validating...");

            List<double> acc_train = new List<double>();
            List<double> acc_test = new List<double>();
            
            for (int i = 0; i < n_folds; i++)
            {
                // Collect training and test data from folds
                var idx_test = idx_folds[i];
                var idx_train = idx_all.Delete(idx_test);
                var X_train = X[idx_train];
                var y_train = y[idx_train];
                var X_test = X[idx_test];
                var y_test = y[idx_test];

                // Build neural network classifier model and train
                NetworkModel model = new NetworkModel();

                model.Layers.Add(new NeuralLayer(d, "INPUT"));
                model.Layers.Add(new NeuralLayer(n_hidden_nodes, "HIDDEN"));
                model.Layers.Add(new NeuralLayer(nClasses, "OUTPUT"));

                model.Build();

                model.Train(X_train, y_train, iterations: n_epochs, learningRate: l_rate);

                // Make predictions for training and test data
                var y_train_predict = model.Predict(X_train);
                var y_test_predict = model.Predict(X_test);

                acc_train.Add(100 * y_train.Sum(y_train_predict) / y_train.Shape[0]);
                acc_test.Add(100 * y_test.Sum(y_test_predict) / y_test.Shape[0]);
                
                Console.WriteLine($"Fold {i + 1}/{n_folds}: train acc = {acc_train.Last()}%, test acc = {acc_test.Last()}% (n_train = {X_train.Shape[0]}, n_test = {X_test.Shape[0]})");
            }

            Console.WriteLine($"Avg train acc = {acc_train.Average()}%, avg test acc = {acc_test.Average()}%");
        }
    }
}
