using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NumSharp.Extensions;
using NumSharp;

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
            var N = X.Shape.Rows; var d = X.Shape.Dimensions; 

            var nClasses = y.Unique().Length;

            Console.WriteLine($" X.shape = {X.Shape}");
            Console.WriteLine($" y.shape = {y.Shape}");
            Console.WriteLine($" n_classes = {nClasses}");

            /* ===================================
               Create cross-validation folds
               These are a list of a list of indices for each fold
              =================================== */

            var idx_all = np.ARange(N, 0);
        }
    }
}
