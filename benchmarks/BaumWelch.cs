#:project ../Bigtree.Algorithm/Bigtree.Algorithm.csproj
#:property PublishAot=false
#:property NoWarn=NDW012;NDW016;NDW017;CS8632
// Baum-Welch: Bigtree 0.4.0's log-space Run (verbatim below, as LegacyBaumWelch) against 0.5.0's Run (Rabiner's scaled
// forward-backward) on the same models and sequences, 10 iterations each; both warmed, paused for the tier-1 JIT, warmed
// again, then 10 interleaved rounds, best of each.
//   dotnet run -c Release benchmarks/BaumWelch.cs
using System.Diagnostics;
using Bigtree.Algorithm.HiddenMarkovModel;
using Bigtree.Algorithm.HiddenMarkovModel.Helpers;
using Bigtree.Algorithm.HiddenMarkovModel.MathHelpers;
using Bigtree.Algorithm.HiddenMarkovModel.Learning.Unsupervised;
using HMM = Bigtree.Algorithm.HiddenMarkovModel.HiddenMarkovModel;

HMM Start(int n, int m, int seed)
{
    var r = new Random(seed);
    double[,] Rows(int a, int b) { var x = new double[a, b]; for (int i = 0; i < a; i++) { double s = 0; for (int j = 0; j < b; j++) s += x[i, j] = 0.5 + r.NextDouble(); for (int j = 0; j < b; j++) x[i, j] /= s; } return x; }
    var pi = new double[n]; for (int i = 0; i < n; i++) pi[i] = 1.0 / n;
    return new HMM(Rows(n, n), Rows(n, m), pi);
}
int[][] Sequences(int k, int t, int m, int seed) { var r = new Random(seed); return Enumerable.Range(0, k).Select(_ => Enumerable.Range(0, t).Select(__ => r.Next(m)).ToArray()).ToArray(); }

foreach (var (K, T, N, M) in new[] { (10, 50, 2, 3), (20, 50, 3, 4), (20, 100, 3, 4), (40, 100, 3, 4), (20, 100, 5, 6), (80, 100, 3, 4), (40, 100, 4, 6), (20, 200, 6, 8), (100, 200, 8, 16) })
{
    var data = Sequences(K, T, M, 1);
    double Once(bool legacy)
    {
        var model = Start(N, M, 2);
        var sw = Stopwatch.StartNew();
        if (legacy) new LegacyBaumWelch(model) { Iterations = 10, Tolerance = double.NegativeInfinity }.Run(data);
        else new BaumWelchLearning(model) { Iterations = 10, Tolerance = double.NegativeInfinity }.Run(data);
        return sw.Elapsed.TotalMilliseconds;
    }
    // Warm both, pause so the tier-1 compiler catches up, warm again (one warm-up pass only QUEUES tier-1).
    for (int w = 0; w < 4; w++) { Once(true); Once(false); }
    Thread.Sleep(300);
    for (int w = 0; w < 4; w++) { Once(true); Once(false); }
    double bestLegacy = double.MaxValue, bestCurrent = double.MaxValue;
    for (int round = 0; round < 10; round++) { bestLegacy = Math.Min(bestLegacy, Once(true)); bestCurrent = Math.Min(bestCurrent, Once(false)); }
    Console.WriteLine($"K={K,4} T={T,5} N={N,3} M={M,3} rows*N^2={K * T * N * N,9}: 0.4.0 {bestLegacy,9:F2} ms | 0.5.0 {bestCurrent,9:F2} ms | x{bestLegacy / bestCurrent,5:F2}");
}

/// <summary>Bigtree 0.4.0's discrete Baum-Welch Run, verbatim (commit dfb20dd), as the benchmark's baseline.</summary>
public class LegacyBaumWelch
{
    /// <summary>The log weight of every sequence.</summary>
    protected double[] mLogWeights;

    /// <summary>Per-sequence log state posteriors.</summary>
    protected double[][,] mLogGamma;

    /// <summary>Per-sequence log transition posteriors.</summary>
    protected double[][][,] mLogKsi;

    /// <summary>The model trained in place.</summary>
    protected HMM mModel;

    /// <summary>The convergence threshold.</summary>
    protected double mTolerance = 0.0001;

    /// <summary>The iteration limit (0 = until convergence).</summary>
    protected int mIterations = 0;

    /// <summary>The convergence threshold on the log-likelihood improvement.</summary>
    public double Tolerance { get { return mTolerance; } set { mTolerance = value; } }

    /// <summary>The iteration limit.</summary>
    public int Iterations { get { return mIterations; } set { mIterations = value; } }

    /// <summary>Creates the baseline learner for a model.</summary>
    /// <param name="hmm">The model to train in place.</param>
    public LegacyBaumWelch(HMM hmm) { mModel = hmm; }
        /// <summary>0.4.0's unweighted Run.</summary>
        /// <param name="observations_db">The observation sequences.</param>
        /// <returns>0.4.0's convergence value (the last sequence's log-likelihood over K: its bug, kept).</returns>
        public double Run(int[][] observations_db)
        {
            return Run(observations_db, null);
        }

        /// <summary>0.4.0's Run, verbatim: scalar forward/backward, E-step and M-step.</summary>
        /// <param name="observations_db">The observation sequences.</param>
        /// <param name="weights">Optional sequence weights.</param>
        /// <returns>0.4.0's convergence value.</returns>
        public double Run(int[][] observations_db, double[] weights)
        {
            ValidationHelper.ValidateObservationDb(observations_db, 0, mModel.SymbolCount);

            int K = observations_db.Length;

            mLogWeights = new double[K];
            if (weights != null)
            {
                for (int k = 0; k < K; ++k)
                {
                    mLogWeights[k] = System.Math.Log(weights[k]);
                }
            }

            int N = mModel.StateCount;
            double lnK = System.Math.Log(K);

            double[,] logA = mModel.LogTransitionMatrix;
            double[,] logB = mModel.LogEmissionMatrix;
            double[] logPi = mModel.LogProbabilityVector;

            int M = mModel.SymbolCount;

            mLogGamma = new double[K][,];
            mLogKsi = new double[K][][,];

            for (int k = 0; k < K; ++k)
            {
                int T = observations_db[k].Length;
                mLogGamma[k] = new double[T, N];
                mLogKsi[k] = new double[T][,];

                for (int t = 0; t < T; ++t)
                {
                    mLogKsi[k][t] = new double[N, N];
                }
            }

            int maxT = observations_db.Max(x => x.Length);
            double[,] lnfwd = new double[maxT, N];
            double[,] lnbwd = new double[maxT, N];

            // Initialize the model log-likelihoods
            double newLogLikelihood = Double.NegativeInfinity;
            double oldLogLikelihood = Double.NegativeInfinity;

            int iteration = 0;
            double deltaLogLikelihood = 0;
            bool should_continue = true;

            do // Until convergence or max iterations is reached
            {
                oldLogLikelihood = newLogLikelihood;

                for (int k = 0; k < K; ++k)
                {
                    int[] observations = observations_db[k];
                    double[,] logGamma = mLogGamma[k];
                    double[][,] logKsi = mLogKsi[k];
                    double w = mLogWeights[k];
                    int T = observations.Length;

                    ForwardBackwardAlgorithm.LogForward(logA, logB, logPi, observations, lnfwd);
                    ForwardBackwardAlgorithm.LogBackward(logA, logB, logPi, observations, lnbwd);

                    // Compute Gamma values
                    for (int t = 0; t < T; ++t)
                    {
                        double lnsum = double.NegativeInfinity;
                        for (int i = 0; i < N; ++i)
                        {
                            logGamma[t, i] = lnfwd[t, i] + lnbwd[t, i] + w;
                            lnsum = LogHelper.LogSum(lnsum, logGamma[t, i]);
                        }
                        if (lnsum != Double.NegativeInfinity)
                        {
                            for (int i = 0; i < N; ++i)
                            {
                                logGamma[t, i] = logGamma[t, i] - lnsum;
                            }
                        }
                    }

                    // Compute Ksi values
                    for (int t = 0; t < T-1; ++t)
                    {
                        double lnsum = double.NegativeInfinity;
                        int x = observations[t + 1];

                        for (int i = 0; i < N; ++i)
                        {
                            for (int j = 0; j < N; ++j)
                            {
                                logKsi[t][i, j] = lnfwd[t, i] + logA[i, j] + lnbwd[t + 1, j] + logB[j, x] + w;
                                lnsum = LogHelper.LogSum(lnsum, logKsi[t][i, j]);
                            }
                        }

                        for(int i=0; i < N; ++i)
                        {
                            for(int j=0; j < N; ++j)
                            {
                                logKsi[t][i, j]=logKsi[t][i, j]-lnsum;
                            }
                        }
                           
                    }

                    newLogLikelihood = Double.NegativeInfinity;
                    for (int i = 0; i < N; ++i)
                    {
                        newLogLikelihood = LogHelper.LogSum(newLogLikelihood, lnfwd[T - 1, i]);
                    }
                }
                
                newLogLikelihood /= K;

                deltaLogLikelihood = newLogLikelihood - oldLogLikelihood;

                iteration++;

                //Console.WriteLine("Iteration: {0}", iteration);

                if (ShouldTerminate(deltaLogLikelihood, iteration))
                {
                    should_continue = false;
                }
                else
                {
                    // update pi
                    for (int i = 0; i < N; ++i)
                    {
                        double lnsum = double.NegativeInfinity;
                        for (int k = 0; k < K; ++k)
                        {
                            lnsum = LogHelper.LogSum(lnsum, mLogGamma[k][0, i]);
                        }
                        logPi[i] = lnsum - lnK;
                    }

                    // update A
                    for (int i = 0; i < N; ++i)
                    {
                        for (int j = 0; j < N; ++j)
                        {
                            double lndenom = double.NegativeInfinity;
                            double lnnum = double.NegativeInfinity;

                            for (int k = 0; k < K; ++k)
                            {
                                
                                int T = observations_db[k].Length;

                                for (int t = 0; t < T - 1; ++t)
                                {
                                    lnnum = LogHelper.LogSum(lnnum, mLogKsi[k][t][i, j]);
                                    lndenom = LogHelper.LogSum(lndenom, mLogGamma[k][t, i]);
                                }

                            }
                            
                            logA[i, j] = (lnnum == lndenom) ? 0 : lnnum - lndenom;
                        }
                    }

                    // update B
                    for (int i = 0; i < N; ++i)
                    {
                        for (int m = 0; m < M; ++m)
                        {
                            double lndenom = double.NegativeInfinity;
                            double lnnum = double.NegativeInfinity;

                            for (int k = 0; k < K; ++k)
                            {   
                                int[] observations=observations_db[k];
                                int T=observations.Length;

                                for (int t = 0; t < T; ++t)
                                {
                                    lndenom = LogHelper.LogSum(lndenom, mLogGamma[k][t, i]);

                                    if (observations[t] == m)
                                    {
                                        lnnum = LogHelper.LogSum(lnnum, mLogGamma[k][t, i]);
                                    }
                                }
                            }
                            logB[i, m] = lnnum-lndenom;
                        }
                    }
                }
            } while (should_continue);

            return newLogLikelihood;
        }

        /// <summary>0.4.0's stopping rule.</summary>
        /// <param name="change">The log-likelihood improvement.</param>
        /// <param name="iteration">The iterations done.</param>
        /// <returns><c>true</c> to stop.</returns>
        protected bool ShouldTerminate(double change, int iteration)
        {
            if (change <= mTolerance)
            {
                return true;
            }

            if (mIterations > 0 && mIterations <= iteration)
            {
                return true;
            }

            return false;
        }
}
