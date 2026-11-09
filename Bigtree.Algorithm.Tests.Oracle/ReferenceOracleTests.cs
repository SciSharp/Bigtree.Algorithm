using System;
using System.IO;
using System.Linq;
using Bigtree.Algorithm.DecisionTree;
using Bigtree.Algorithm.HiddenMarkovModel.Learning.Unsupervised;
using Bigtree.Algorithm.NeuralNetwork;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;
using HMM = Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel;

namespace Bigtree.Algorithm.Tests.Oracle
{
    /// <summary>
    ///     The algorithms scikit-learn has no twin for, against NumPy / SciPy reference implementations written here
    ///     from the textbook: the neural network, Baum-Welch, the ID3 gains, and the cross-validation folds.
    /// </summary>
    [TestClass]
    public class ReferenceOracleTests : OracleTest
    {
        /// <summary>
        ///     The seeds network (7-5-3, mini-batches of 8, 30 epochs) from the same initial weights: NumPy's
        ///     mini-batch backpropagation ends at the same weights to 1e-12.
        /// </summary>
        [TestMethod]
        [NDScoped]
        public void Network_MatchesNumPyBackpropagation()
        {
            var (X, y) = Utils.ReadCsv(Path.Combine(AppContext.BaseDirectory, "data", "seeds_dataset.csv"));
            NDArray scaled = Utils.Normalize(X);
            using var model = new NetworkModel();
            model.AddLayer(new NeuralLayer(7));
            model.AddLayer(new NeuralLayer(5));
            model.AddLayer(new NeuralLayer(3));
            model.Build(seed: 3);

            using (Py.GIL())
            {
                py.X = scaled;
                py.y = y;
                py.W0 = model.Weights[0].copy();
                py.W1 = model.Weights[1].copy();
                py.Exec(
                    "X = np.array(X); y = np.array(y); W = [np.array(W0), np.array(W1)]\n" +
                    "T = np.eye(3)[y]\n" +
                    "sig = lambda z: 1.0 / (1.0 + np.exp(-z))\n" +
                    "for epoch in range(30):\n" +
                    "    for s in range(0, len(X), 8):\n" +
                    "        a = [X[s:s + 8]]\n" +
                    "        for w in W: a.append(sig(a[-1] @ w.T))\n" +
                    "        d = (T[s:s + 8] - a[-1]) * a[-1] * (1 - a[-1])\n" +
                    "        for l in (1, 0):\n" +
                    "            prev = (d @ W[l]) * a[l] * (1 - a[l]) if l > 0 else None\n" +
                    "            W[l] = W[l] + (1.5 / len(a[0])) * (d.T @ a[l])\n" +
                    "            d = prev\n" +
                    "R0, R1 = W");
                model.Train(scaled, y, iterations: 30, learningRate: 1.5, batchSize: 8);

                AssertClose(py.R0, model.Weights[0], 1e-12, 1e-12, "input -> hidden weights");
                AssertClose(py.R1, model.Weights[1], 1e-12, 1e-12, "hidden -> output weights");
            }
        }

        /// <summary>
        ///     Baum-Welch, 6 iterations from the same start: Run's scaled forward-backward against a NumPy / SciPy
        ///     log-space implementation, which learns the same pi, A and B to 1e-9.
        /// </summary>
        /// <param name="sequences">The number of observation sequences.</param>
        /// <param name="length">The length of every sequence.</param>
        [TestMethod]
        [DataRow(8, 40, DisplayName = "8 x 40")]
        [DataRow(80, 100, DisplayName = "80 x 100")]
        [NDScoped]
        public void BaumWelch_MatchesNumPyReEstimation(int sequences, int length)
        {
            var random = new Random(9);
            int[][] data = Enumerable.Range(0, sequences)
                .Select(_ => Enumerable.Range(0, length).Select(__ => random.Next(3)).ToArray()).ToArray();
            double[,] A = { { 0.7, 0.3 }, { 0.4, 0.6 } }, B = { { 0.5, 0.3, 0.2 }, { 0.2, 0.3, 0.5 } };
            double[] pi = { 0.55, 0.45 };
            var model = new HMM(A, B, pi);
            new BaumWelchLearning(model) { Iterations = 6, Tolerance = double.NegativeInfinity }.Run(data);

            using (Py.GIL())
            {
                py.obs = np.array(data.SelectMany(s => s).ToArray()).reshape(sequences, length);
                py.Exec(
                    "from scipy.special import logsumexp\n" +
                    "obs = np.array(obs)\n" +
                    "lA = np.log([[0.7, 0.3], [0.4, 0.6]]); lB = np.log([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]); lpi = np.log([0.55, 0.45])\n" +
                    "for it in range(5):\n" +
                    "    G, KS = [], []\n" +
                    "    for o in obs:\n" +
                    "        T = len(o); f = np.empty((T, 2)); b = np.zeros((T, 2))\n" +
                    "        f[0] = lpi + lB[:, o[0]]\n" +
                    "        for t in range(1, T): f[t] = logsumexp(f[t - 1][:, None] + lA, axis=0) + lB[:, o[t]]\n" +
                    "        for t in range(T - 2, -1, -1): b[t] = logsumexp(lA + (lB[:, o[t + 1]] + b[t + 1])[None, :], axis=1)\n" +
                    "        g = f + b; g -= logsumexp(g, axis=1)[:, None]; G.append(g)\n" +
                    "        k = f[:-1, :, None] + lA[None] + (b[1:] + lB[:, o[1:]].T)[:, None, :]\n" +
                    "        KS.append(k - logsumexp(k, axis=(1, 2))[:, None, None])\n" +
                    "    lpi = logsumexp(np.stack([g[0] for g in G]), axis=0) - np.log(len(obs))\n" +
                    "    num = logsumexp(np.concatenate(KS), axis=0); den = logsumexp(np.concatenate([g[:-1] for g in G]), axis=0)\n" +
                    "    lA = num - den[:, None]\n" +
                    "    allg = np.concatenate(G); allo = obs.ravel()\n" +
                    "    lB = np.stack([logsumexp(allg[allo == m], axis=0) for m in range(3)], axis=1) - logsumexp(allg, axis=0)[:, None]");

                AssertClose(py.lpi, np.array(model.LogProbabilityVector), 1e-9, 1e-9, "log pi");
                AssertClose(py.lA, Matrix(model.LogTransitionMatrix), 1e-9, 1e-9, "log A");
                AssertClose(py.lB, Matrix(model.LogEmissionMatrix), 1e-9, 1e-9, "log B");
            }
        }

        /// <summary>
        ///     The ID3 information gains of the NeedUmbrella table's attributes equal SciPy's: H(S) - sum over values of
        ///     |S_v| / |S| H(S_v), with <c>scipy.stats.entropy(counts, base=2)</c>.
        /// </summary>
        [TestMethod]
        public void DecisionTree_GainsMatchSciPyEntropy()
        {
            var data = new DataReader().Read(Path.Combine(AppContext.BaseDirectory, "data", "DecisionTree.csv"));
            TreeNode root = Tree.Learn(data, "");
            string[] attributes = data.Columns.Take(data.Columns.Length - 1).ToArray();

            using (Py.GIL())
            {
                // Module-level names only: a function defined by Exec looks its globals up in the module, not in a
                // locals dictionary passed beside the code.
                py.path = Path.Combine(AppContext.BaseDirectory, "data", "DecisionTree.csv");
                py.Exec(
                    "from scipy.stats import entropy\n" +
                    "rows = [line.rstrip(';').split(';') for line in open(path).read().splitlines()[1:] if line]\n" +
                    "labels = [r[-1] for r in rows]\n" +
                    "def H(subset): return entropy([subset.count(v) for v in sorted(set(subset))], base=2)\n" +
                    "gains = []\n" +
                    "for c in range(len(rows[0]) - 1):\n" +
                    "    values = [r[c] for r in rows]\n" +
                    "    rest = sum(values.count(v) / len(rows) * H([l for x, l in zip(values, labels) if x == v]) for v in set(values))\n" +
                    "    gains.append(H(labels) - rest)\n" +
                    "gains = np.array(gains); best = int(np.argmax(gains))");
                Assert.AreEqual(attributes[(int)py.best], root.Name, "the root is the attribute SciPy ranks first");
                Assert.AreEqual((double)py.gains[0], root.NodeAttribute.InformationGain, 1e-12, "the root's gain");
            }
        }

        /// <summary>The cross-validation folds are slices of NumPy's <c>default_rng(seed).permutation(N)</c>, element for element.</summary>
        [TestMethod]
        [NDScoped]
        public void CrossValFolds_AreNumPysPermutation()
        {
            NDArray[] folds = Utils.CrossValFolds(210, 4, seed: 123);
            using (Py.GIL())
            {
                py.Exec("perm = np.random.default_rng(123).permutation(210)");
                for (int i = 0; i < 4; i++)
                {
                    py.i = i;
                    py.Exec("fold = perm[i * 52:(i + 1) * 52]");
                    AssertEqual(py.fold, folds[i], $"fold {i}");
                }
            }
        }

        /// <summary>A 2-D managed array as a NumSharp array of the same shape.</summary>
        /// <param name="matrix">The array.</param>
        /// <returns>The copy.</returns>
        private static NDArray Matrix(double[,] matrix)
            => np.array(matrix.Cast<double>().ToArray()).reshape(matrix.GetLength(0), matrix.GetLength(1));
    }
}
