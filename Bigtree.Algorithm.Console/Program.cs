using Bigtree.Algorithm.Bayes;
using Bigtree.Algorithm.KNN;
using Bigtree.Algorithm.NeuralNetwork;
using Bigtree.Algorithm.SVM;
using NumSharp;
using System;
using System.Diagnostics;
using System.IO;

namespace Bigtree.Algorithm.Console
{
    /// <summary>
    ///     A tour of the NumSharp estimators on the UCI seeds dataset: every model is fitted on the same 80% and scored
    ///     on the same held-out 20%.
    /// </summary>
    class Program
    {
        /// <summary>Loads the data, fits each model, prints its held-out accuracy and fit time.</summary>
        /// <param name="args">Unused.</param>
        static void Main(string[] args)
        {
            using var scope = NDScope.Open();
            var (X, y) = Utils.ReadCsv(Path.Combine(AppContext.BaseDirectory, "data", "seeds_dataset.csv"));
            NDArray scaled = Utils.Normalize(X);
            NDArray test = Utils.CrossValFolds((int)X.shape[0], 5, seed: 0)[0];
            NDArray train = np.setdiff1d(np.arange(X.shape[0]), test);
            NDArray Xtrain = scaled[train], ytrain = y[train], Xtest = scaled[test], ytest = y[test];

            System.Console.WriteLine($"seeds: {X.shape[0]} samples, {X.shape[1]} features, {np.unique(y).values.size} classes; train {train.size}, test {test.size}");

            Report("Gaussian Naive Bayes", () => new GaussianNaiveBayes().Fit(Xtrain, ytrain).Score(Xtest, ytest));
            Report("Bernoulli Naive Bayes (binarize at 0.5)", () => new BernoulliNaiveBayes(binarize: 0.5).Fit(Xtrain, ytrain).Score(Xtest, ytest));
            Report("KNN (k = 5)", () => new KnnModel(k: 5).Fit(Xtrain, ytrain).Score(Xtest, ytest));
            Report("Neural network 7-5-3 (1000 epochs, batch 8)", () =>
            {
                using var model = new NetworkModel();
                model.AddLayer(new NeuralLayer(7, "INPUT"));
                model.AddLayer(new NeuralLayer(5, "HIDDEN"));
                model.AddLayer(new NeuralLayer(3, "OUTPUT"));
                model.Build(seed: 0);
                model.Train(Xtrain, ytrain, iterations: 1000, learningRate: 1.5, batchSize: 8);
                return np.mean((model.Predict(Xtest) == ytest).astype(np.float64)).GetDouble();
            });
            Report("SVM (C-SVC, RBF)", () =>
            {
                Model model = Training.Train(Problem.FromNDArray(Xtrain, ytrain), new Parameter { Gamma = 1.0 / 7 });
                return np.mean((model.PredictLabels(Xtest) == ytest.astype(np.float64)).astype(np.float64)).GetDouble();
            });
        }

        /// <summary>Times one fit-and-score and prints the accuracy.</summary>
        /// <param name="name">The model's description.</param>
        /// <param name="fitAndScore">Fits the model and returns its held-out accuracy.</param>
        static void Report(string name, Func<double> fitAndScore)
        {
            var clock = Stopwatch.StartNew();
            double accuracy = fitAndScore();
            System.Console.WriteLine($"  {name,-46} accuracy {accuracy:P1}   ({clock.Elapsed.TotalMilliseconds:F0} ms)");
        }
    }
}
