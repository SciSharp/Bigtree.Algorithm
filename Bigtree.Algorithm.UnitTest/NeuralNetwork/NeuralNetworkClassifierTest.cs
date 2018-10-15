using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NumSharp.Extensions;
using NumSharp;
using Bigtree.Algorithm.NeuralNetwork;

namespace Bigtree.Algorithm.UnitTest.NeuralNetwork
{
    [TestClass]
    public class NeuralNetworkClassifierTest
    {
        [TestMethod]
        public void SeedsClassificationTest()
        {
            var filename = @"data/seeds_dataset.csv";
            var np = new NdArray<int>();

            /* =================================
                Read data (X,y) and normalize X
               ================================= */
            Console.WriteLine($"Reading '{filename}'...");
            (var X, var y) = Utils.ReadCsv<double, int>(filename); // read as matrix of floats and int
            // normalize
            X.Normalize(); 
            // extract shape of X
            (var N, var d) = X.Shape; 

            var nClasses = y.Unique().Length;

            Console.WriteLine($" X.shape = {X.Shape}");
            Console.WriteLine($" y.shape = {y.Shape}");
            Console.WriteLine($" n_classes = {nClasses}");

            /* ===================================
               Create cross-validation folds
               These are a list of a list of indices for each fold
              =================================== */

            var idx_all = np.ARange(N, 0);
            var idx_folds = Utils.CrossValFolds(N, 4);

            /* ===================================
              Train and evaluate the model on each fold
              =================================== */
            // List<acc_train, acc_test = list(), list()  # training/test accuracy score
            Console.WriteLine("Training and cross-validating...");
            for(int i = 0; i < 4; i++)
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

                model.Layers.Add(new NeuralLayer(d, 0.1, "INPUT"));
                model.Layers.Add(new NeuralLayer(5, 0.1, "HIDDEN"));
                model.Layers.Add(new NeuralLayer(nClasses, 0.1, "OUTPUT"));

                model.Build();

                model.Train(X_train, y_train, iterations: 800, learningRate: 0.6);

                // Make predictions for training and test data
                var y_train_predict = model.Predict(X_train);
                var y_test_predict = model.Predict(X_test);

                var acc_train = 100 * y_train.Sum(y_train_predict) / y_train.Length;
                var acc_test = 100 * y_test.Sum(y_test_predict) / y_test.Length;
            }
        }
    }
}
