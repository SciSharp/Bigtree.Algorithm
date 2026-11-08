using Bigtree.Algorithm.HiddenMarkovModel;
using Bigtree.Algorithm.HiddenMarkovModel.Learning.Unsupervised;
using Bigtree.Algorithm.HiddenMarkovModel.MathHelpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest.HiddenMarkovModel
{
    /// <summary>
    ///     The discrete HMM: the vectorized Baum-Welch against Bigtree 0.4.0's scalar re-estimation, the EM guarantee
    ///     (the likelihood never decreases), and the scaled forward/backward pair that 0.5.0 repaired.
    /// </summary>
    [TestClass]
    public class BaumWelchTests
    {
        /// <summary>The model that generates the test sequences: two sticky states with distinct emissions.</summary>
        private static readonly double[,] TrueA = { { 0.85, 0.15 }, { 0.25, 0.75 } };

        /// <summary>The generating model's emissions over three symbols.</summary>
        private static readonly double[,] TrueB = { { 0.7, 0.2, 0.1 }, { 0.1, 0.3, 0.6 } };

        /// <summary>The generating model's initial distribution.</summary>
        private static readonly double[] TruePi = { 0.6, 0.4 };

        /// <summary>
        ///     After the same number of iterations from the same start, Run and 0.4.0's scalar re-estimation learn the
        ///     same probabilities. 750 observations × 2² states run the scalar path; 7,200 × 2² (28,800, above the
        ///     25,000 gate) the NumSharp E/M step, which matches to rounding (its sums run in another order).
        /// </summary>
        /// <param name="count">The number of sequences.</param>
        /// <param name="length">The length of every sequence.</param>
        [TestMethod]
        [DataRow(25, 30, DisplayName = "scalar path")]
        [DataRow(60, 120, DisplayName = "NumSharp path")]
        public void MatchesTheScalarReEstimation(int count, int length)
        {
            int[][] sequences = Sample(new Random(1), count, length);
            var trained = StartModel();
            var scalar = StartModel();

            new BaumWelchLearning(trained) { Iterations = 8, Tolerance = double.NegativeInfinity }.Run(sequences);
            ScalarBaumWelch(scalar, sequences, mSteps: 7);

            AssertClose(scalar.LogProbabilityVector, trained.LogProbabilityVector);
            AssertClose(Flatten(scalar.LogTransitionMatrix), Flatten(trained.LogTransitionMatrix));
            AssertClose(Flatten(scalar.LogEmissionMatrix), Flatten(trained.LogEmissionMatrix));
        }

        /// <summary>Sequence weights shift the re-estimation exactly as in 0.4.0, on both paths.</summary>
        /// <param name="count">The number of sequences.</param>
        /// <param name="length">The length of every sequence.</param>
        [TestMethod]
        [DataRow(6, 12, DisplayName = "scalar path")]
        [DataRow(60, 120, DisplayName = "NumSharp path")]
        public void WeightedSequencesMatchTheScalarReEstimation(int count, int length)
        {
            int[][] sequences = Sample(new Random(2), count, length);
            double[] weights = Enumerable.Range(0, count).Select(k => 0.5 + (k % 4) * 0.75).ToArray();
            var trained = StartModel();
            var scalar = StartModel();

            new BaumWelchLearning(trained) { Iterations = 4, Tolerance = double.NegativeInfinity }.Run(sequences, weights);
            ScalarBaumWelch(scalar, sequences, mSteps: 3, weights);

            AssertClose(Flatten(scalar.LogTransitionMatrix), Flatten(trained.LogTransitionMatrix));
            AssertClose(Flatten(scalar.LogEmissionMatrix), Flatten(trained.LogEmissionMatrix));
        }

        /// <summary>
        ///     EM never lowers the likelihood: the total log-likelihood of the data under the model is non-decreasing
        ///     from one iteration count to the next, and the trained model explains the data better than its start.
        /// </summary>
        [TestMethod]
        public void LikelihoodNeverDecreases()
        {
            int[][] sequences = Sample(new Random(3), count: 20, length: 40);
            double previous = TotalLogLikelihood(StartModel(), sequences);
            for (int iterations = 2; iterations <= 12; iterations += 2)
            {
                var model = StartModel();
                new BaumWelchLearning(model) { Iterations = iterations, Tolerance = double.NegativeInfinity }.Run(sequences);
                double current = TotalLogLikelihood(model, sequences);
                Assert.IsTrue(current >= previous - 1e-9, $"{iterations} iterations: {current} < {previous}");
                previous = current;
            }
        }

        /// <summary>
        ///     The convergence value is log(sum of the sequences' likelihoods) / K, over every sequence (0.4.0 measured
        ///     only the last sequence).
        /// </summary>
        [TestMethod]
        public void ConvergenceValueCoversEverySequence()
        {
            int[][] sequences = Sample(new Random(4), count: 5, length: 10);
            var model = StartModel();
            double reported = new BaumWelchLearning(model) { Iterations = 1, Tolerance = double.NegativeInfinity }.Run(sequences);

            // Iterations = 1 stops before the first update, so the reported value is the start model's.
            double sum = double.NegativeInfinity;
            var start = StartModel();
            foreach (int[] sequence in sequences)
                sum = LogHelper.LogSum(sum, start.Evaluate(sequence));
            Assert.AreEqual(sum / sequences.Length, reported, 1e-10);
        }

        /// <summary>Sequences of one observation have no transitions; training still runs.</summary>
        [TestMethod]
        public void SingleObservationSequences()
        {
            var model = StartModel();
            double result = new BaumWelchLearning(model) { Iterations = 3, Tolerance = double.NegativeInfinity }
                .Run(new[] { new[] { 0 }, new[] { 2 }, new[] { 1 } });
            Assert.IsFalse(double.IsNaN(result));
            Assert.IsTrue(model.LogProbabilityVector.All(p => !double.IsNaN(p)));
        }

        /// <summary>No sequences, an empty one, mismatched weights and out-of-range symbols are refused.</summary>
        [TestMethod]
        public void RefusesBadInput()
        {
            var learner = new BaumWelchLearning(StartModel());
            Assert.ThrowsException<ArgumentException>(() => learner.Run(new int[0][]));
            Assert.ThrowsException<ArgumentException>(() => learner.Run(new[] { new int[0] }));
            Assert.ThrowsException<ArgumentException>(() => learner.Run(new[] { new[] { 0 } }, new[] { 1.0, 2.0 }));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => learner.Run(new[] { new[] { 3 } }));
        }

        /// <summary>
        ///     The scaled forward pass now records its first scale, so its log-likelihood equals the log-space forward
        ///     pass's (it was -infinity), and the scaled backward pass terminates (its loop incremented N) with finite
        ///     values that satisfy sum_i fwd[0, i] * bwd[0, i] * c_0 = 1.
        /// </summary>
        [TestMethod]
        public void ScaledForwardBackward()
        {
            int[] observations = { 0, 2, 2, 1, 0, 0, 2 };
            new ForwardBackwardAlgorithm().Forward(TrueA, TrueB, TruePi, observations, out double[] scale, out double logLikelihood);
            ForwardBackwardAlgorithm.LogForward(LogHelper.Log(TrueA), LogHelper.Log(TrueB), LogHelper.Log(TruePi), observations, out double expected);
            Assert.AreEqual(expected, logLikelihood, 1e-12);

            double[,] fwd = new ForwardBackwardAlgorithm().Forward(TrueA, TrueB, TruePi, observations, out scale);
            double[,] bwd = ForwardBackwardAlgorithm.Backward(TrueA, TrueB, TruePi, observations, scale);
            double total = 0;
            for (int i = 0; i < 2; i++)
                total += fwd[0, i] * bwd[0, i] * scale[0];
            Assert.AreEqual(1.0, total, 1e-12);
        }

        /// <summary>A deliberately wrong starting model (both states alike), so training has work to do.</summary>
        /// <returns>A fresh model.</returns>
        private static Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel StartModel()
            => new Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel(
                new[,] { { 0.6, 0.4 }, { 0.45, 0.55 } },
                new[,] { { 0.4, 0.35, 0.25 }, { 0.3, 0.3, 0.4 } },
                new[] { 0.5, 0.5 });

        /// <summary>Samples observation sequences from the generating model.</summary>
        /// <param name="random">The random source.</param>
        /// <param name="count">The number of sequences.</param>
        /// <param name="length">The length of every sequence.</param>
        /// <returns>The sequences.</returns>
        private static int[][] Sample(Random random, int count, int length)
        {
            int Draw(Func<int, double> probability, int outcomes)
            {
                double u = random.NextDouble(), sum = 0;
                for (int i = 0; i < outcomes - 1; i++)
                    if (u < (sum += probability(i)))
                        return i;
                return outcomes - 1;
            }

            var sequences = new int[count][];
            for (int k = 0; k < count; k++)
            {
                sequences[k] = new int[length];
                int state = Draw(i => TruePi[i], 2);
                for (int t = 0; t < length; t++)
                {
                    sequences[k][t] = Draw(m => TrueB[state, m], 3);
                    int from = state;
                    state = Draw(j => TrueA[from, j], 2);
                }
            }
            return sequences;
        }

        /// <summary>The summed log-likelihood of the sequences under a model.</summary>
        /// <param name="model">The model.</param>
        /// <param name="sequences">The sequences.</param>
        /// <returns>sum over sequences of log P(sequence).</returns>
        private static double TotalLogLikelihood(Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel model, int[][] sequences)
            => sequences.Sum(model.Evaluate);

        /// <summary>A 2-D array's elements in row-major order.</summary>
        /// <param name="matrix">The array.</param>
        /// <returns>The flat copy.</returns>
        private static double[] Flatten(double[,] matrix) => matrix.Cast<double>().ToArray();

        /// <summary>Asserts equal lengths and elements equal within 1e-9 (relative to their size).</summary>
        /// <param name="expected">The reference values.</param>
        /// <param name="actual">The values under test.</param>
        private static void AssertClose(double[] expected, double[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], actual[i], 1e-9 * (1 + Math.Abs(expected[i])), $"element {i}");
        }

        /// <summary>
        ///     Bigtree 0.4.0's discrete Baum-Welch re-estimation, scalar and verbatim in its arithmetic (the E-step of
        ///     every sequence, then the pi / A / B updates), run for a fixed number of updates.
        /// </summary>
        /// <param name="model">The model to train in place.</param>
        /// <param name="sequences">The observation sequences.</param>
        /// <param name="mSteps">The number of re-estimations.</param>
        /// <param name="weights">Optional sequence weights.</param>
        private static void ScalarBaumWelch(Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel model, int[][] sequences, int mSteps, double[] weights = null)
        {
            int K = sequences.Length, N = model.StateCount, M = model.SymbolCount;
            double lnK = Math.Log(K);
            double[,] logA = model.LogTransitionMatrix, logB = model.LogEmissionMatrix;
            double[] logPi = model.LogProbabilityVector;
            var logWeights = weights?.Select(w => Math.Log(w)).ToArray() ?? new double[K];
            var gamma = sequences.Select(s => new double[s.Length, N]).ToArray();
            var ksi = sequences.Select(s => Enumerable.Range(0, s.Length).Select(t => new double[N, N]).ToArray()).ToArray();
            int maxT = sequences.Max(s => s.Length);
            var lnfwd = new double[maxT, N];
            var lnbwd = new double[maxT, N];

            for (int step = 0; step < mSteps; step++)
            {
                for (int k = 0; k < K; ++k)
                {
                    int[] observations = sequences[k];
                    int T = observations.Length;
                    double w = logWeights[k];
                    ForwardBackwardAlgorithm.LogForward(logA, logB, logPi, observations, lnfwd);
                    ForwardBackwardAlgorithm.LogBackward(logA, logB, logPi, observations, lnbwd);
                    for (int t = 0; t < T; ++t)
                    {
                        double lnsum = double.NegativeInfinity;
                        for (int i = 0; i < N; ++i)
                        {
                            gamma[k][t, i] = lnfwd[t, i] + lnbwd[t, i] + w;
                            lnsum = LogHelper.LogSum(lnsum, gamma[k][t, i]);
                        }
                        if (lnsum != double.NegativeInfinity)
                            for (int i = 0; i < N; ++i)
                                gamma[k][t, i] = gamma[k][t, i] - lnsum;
                    }
                    for (int t = 0; t < T - 1; ++t)
                    {
                        double lnsum = double.NegativeInfinity;
                        int x = observations[t + 1];
                        for (int i = 0; i < N; ++i)
                            for (int j = 0; j < N; ++j)
                            {
                                ksi[k][t][i, j] = lnfwd[t, i] + logA[i, j] + lnbwd[t + 1, j] + logB[j, x] + w;
                                lnsum = LogHelper.LogSum(lnsum, ksi[k][t][i, j]);
                            }
                        for (int i = 0; i < N; ++i)
                            for (int j = 0; j < N; ++j)
                                ksi[k][t][i, j] = ksi[k][t][i, j] - lnsum;
                    }
                }

                for (int i = 0; i < N; ++i)
                {
                    double lnsum = double.NegativeInfinity;
                    for (int k = 0; k < K; ++k)
                        lnsum = LogHelper.LogSum(lnsum, gamma[k][0, i]);
                    logPi[i] = lnsum - lnK;
                }
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j)
                    {
                        double lndenom = double.NegativeInfinity, lnnum = double.NegativeInfinity;
                        for (int k = 0; k < K; ++k)
                            for (int t = 0; t < sequences[k].Length - 1; ++t)
                            {
                                lnnum = LogHelper.LogSum(lnnum, ksi[k][t][i, j]);
                                lndenom = LogHelper.LogSum(lndenom, gamma[k][t, i]);
                            }
                        logA[i, j] = (lnnum == lndenom) ? 0 : lnnum - lndenom;
                    }
                for (int i = 0; i < N; ++i)
                    for (int m = 0; m < M; ++m)
                    {
                        double lndenom = double.NegativeInfinity, lnnum = double.NegativeInfinity;
                        for (int k = 0; k < K; ++k)
                            for (int t = 0; t < sequences[k].Length; ++t)
                            {
                                lndenom = LogHelper.LogSum(lndenom, gamma[k][t, i]);
                                if (sequences[k][t] == m)
                                    lnnum = LogHelper.LogSum(lnnum, gamma[k][t, i]);
                            }
                        logB[i, m] = lnnum - lndenom;
                    }
            }
        }
    }
}
