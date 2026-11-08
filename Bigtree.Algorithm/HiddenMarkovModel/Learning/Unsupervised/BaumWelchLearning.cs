using Bigtree.Algorithm.HiddenMarkovModel.Helpers;
using Bigtree.Algorithm.HiddenMarkovModel.MathHelpers;
using System;
using System.Linq;

namespace Bigtree.Algorithm.HiddenMarkovModel.Learning.Unsupervised
{
    /// <summary>
    ///     Baum-Welch (expectation-maximization) training of a discrete hidden Markov model from unlabelled observation
    ///     sequences: it re-estimates the initial, transition and emission probabilities until the log-likelihood stops
    ///     improving.
    /// </summary>
    /// <remarks>
    ///     <para><b>How it computes.</b> Rabiner's scaled forward-backward (<c>RunScaled</c>): every time step is
    ///     normalized by its scale <c>c_t</c>, so the recursions are multiply-adds over contiguous rows instead of 0.4.0's
    ///     log-space sums (an <c>exp</c> and a <c>log1p</c> per term), and an iteration's transition statistics
    ///     accumulate as outer products <c>A ∘ Σ α̂_t ⊗ w_t</c> instead of a stored (N, N) posterior per time step. The
    ///     model's parameters stay in log space. Measured against 0.4.0 (10 iterations): 100 sequences × 200
    ///     observations × 8 states 2,842 ms → 52 ms; against 0.5.0's earlier NumSharp E/M step, 400 × 400 × 16 states
    ///     (3 iterations) 15.2 s and a 4.3 GB peak → 0.42 s and no array beyond the sequences' (T × N) buffers. The
    ///     recursions are sequential in time over N-sized vectors, where NumSharp's per-call cost exceeds the work,
    ///     so this class computes without it.</para>
    ///     <para>A sequence that is impossible under the starting model (a step of probability 0) cannot be scaled; then
    ///     the run restarts from the starting model in 0.4.0's log-space loops (<c>RunScalar</c>), which reproduce 0.4.0's
    ///     behaviour for such data.</para>
    ///     <para><b>Changed in 0.5.0.</b> The convergence log-likelihood sums the likelihood of every sequence (in log
    ///     space) before dividing by the sequence count, as in the Accord.NET algorithm this class derives from;
    ///     Bigtree 0.4.0 reset it for every sequence and so measured only the last one. Sequence weights cancel in the
    ///     per-time-step normalization of the posteriors, in 0.4.0 as now, so they do not change the result. The scaled
    ///     path does not fill the per-sequence <see cref="mLogGamma"/> and <see cref="mLogKsi"/> scratch arrays (the
    ///     log-space fallback and the continuous variant still do).</para>
    /// </remarks>
    public partial class BaumWelchLearning : IUnsupervisedLearning
    {
        /// <summary>The log weight of every sequence of the current run (0 when no weights were given).</summary>
        protected double[] mLogWeights;

        /// <summary>Per-sequence log state posteriors; scratch of the continuous variant.</summary>
        protected double[][,] mLogGamma;

        /// <summary>Per-sequence log transition posteriors; scratch of the continuous variant.</summary>
        protected double[][][,] mLogKsi;

        /// <summary>The model being trained; its probability arrays are updated in place.</summary>
        protected HiddenMarkovModel mModel;

        /// <summary>The convergence threshold on the improvement of the log-likelihood.</summary>
        protected double mTolerance = 0.0001;

        /// <summary>The maximum number of iterations; 0 means until convergence.</summary>
        protected int mIterations = 0;

        /// <summary>The model being trained; its probability arrays are updated in place.</summary>
        public HiddenMarkovModel Model
        {
            get { return mModel; }
            set { mModel = value; }
        }

        /// <summary>
        ///     Training stops once an iteration improves the log-likelihood by this much or less (default 0.0001).
        /// </summary>
        public double Tolerance
        {
            get { return mTolerance; }
            set { mTolerance = value; }
        }

        /// <summary>The maximum number of iterations; 0 (the default) iterates until <see cref="Tolerance"/> is met.</summary>
        public int Iterations
        {
            get { return mIterations; }
            set { mIterations = value; }
        }

        /// <summary>Creates a learner that trains <paramref name="hmm"/> in place.</summary>
        /// <param name="hmm">The discrete model to train; its current probabilities are the starting point.</param>
        public BaumWelchLearning(HiddenMarkovModel hmm)
        {
            mModel = hmm;
        }

        /// <summary>Trains the model on unweighted sequences.</summary>
        /// <param name="observations_db">The observation sequences: each a non-empty array of symbols in [0, SymbolCount).</param>
        /// <returns>The final average log-likelihood (see the class remarks).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A symbol is outside [0, SymbolCount).</exception>
        /// <exception cref="ArgumentException">There are no sequences, or one is empty.</exception>
        public double Run(int[][] observations_db)
        {
            return Run(observations_db, null);
        }

        /// <summary>Trains the model on weighted sequences until convergence or <see cref="Iterations"/>.</summary>
        /// <param name="observations_db">The observation sequences: each a non-empty array of symbols in [0, SymbolCount).</param>
        /// <param name="weights">One positive weight per sequence, or <c>null</c> for equal weights.</param>
        /// <returns>
        ///     The final <c>log(sum over sequences of P(sequence)) / K</c>, the quantity whose improvement is compared
        ///     with <see cref="Tolerance"/>.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">A symbol is outside [0, SymbolCount).</exception>
        /// <exception cref="ArgumentException">There are no sequences, one is empty, or the weights do not match the sequences.</exception>
        public double Run(int[][] observations_db, double[] weights)
        {
            ValidationHelper.ValidateObservationDb(observations_db, 0, mModel.SymbolCount);

            int K = observations_db.Length;
            if (K == 0)
                throw new ArgumentException("There are no observation sequences.", nameof(observations_db));
            if (observations_db.Any(sequence => sequence.Length == 0))
                throw new ArgumentException("An observation sequence is empty.", nameof(observations_db));
            if (weights != null && weights.Length != K)
                throw new ArgumentException($"{weights.Length} weights for {K} sequences.", nameof(weights));

            mLogWeights = new double[K];
            if (weights != null)
            {
                for (int k = 0; k < K; ++k)
                {
                    mLogWeights[k] = System.Math.Log(weights[k]);
                }
            }

            // The scaled path updates the model in place; keep the start, in case a sequence turns out impossible under
            // it (a step of probability 0, which only log space represents) and the run has to restart there.
            var startA = (double[,])mModel.LogTransitionMatrix.Clone();
            var startB = (double[,])mModel.LogEmissionMatrix.Clone();
            var startPi = (double[])mModel.LogProbabilityVector.Clone();
            if (RunScaled(observations_db, out double logLikelihood))
                return logLikelihood;
            Buffer.BlockCopy(startA, 0, mModel.LogTransitionMatrix, 0, startA.Length * sizeof(double));
            Buffer.BlockCopy(startB, 0, mModel.LogEmissionMatrix, 0, startB.Length * sizeof(double));
            Buffer.BlockCopy(startPi, 0, mModel.LogProbabilityVector, 0, startPi.Length * sizeof(double));
            return RunScalar(observations_db);
        }

        /// <summary>
        ///     Baum-Welch with Bigtree 0.4.0's log-space loops (one <c>exp</c> and one <c>log1p</c> per summed term): the
        ///     fallback for observation sequences that are impossible under the starting model, which the scaled path
        ///     cannot represent. Fills <see cref="mLogGamma"/> and <see cref="mLogKsi"/> as 0.4.0 did.
        /// </summary>
        /// <param name="observations_db">The validated observation sequences.</param>
        /// <returns>The final average log-likelihood.</returns>
        private double RunScalar(int[][] observations_db)
        {
            int K = observations_db.Length;

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
                // Once per iteration, summed over every sequence below: 0.4.0 reset it inside the sequence loop, so
                // only the last sequence's likelihood was measured.
                newLogLikelihood = Double.NegativeInfinity;

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
                    for (int t = 0; t < T - 1; ++t)
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

                        for (int i = 0; i < N; ++i)
                        {
                            for (int j = 0; j < N; ++j)
                            {
                                logKsi[t][i, j] = logKsi[t][i, j] - lnsum;
                            }
                        }
                    }

                    for (int i = 0; i < N; ++i)
                    {
                        newLogLikelihood = LogHelper.LogSum(newLogLikelihood, lnfwd[T - 1, i]);
                    }
                }

                newLogLikelihood /= K;

                deltaLogLikelihood = newLogLikelihood - oldLogLikelihood;

                iteration++;

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
                                int[] observations = observations_db[k];
                                int T = observations.Length;

                                for (int t = 0; t < T; ++t)
                                {
                                    lndenom = LogHelper.LogSum(lndenom, mLogGamma[k][t, i]);

                                    if (observations[t] == m)
                                    {
                                        lnnum = LogHelper.LogSum(lnnum, mLogGamma[k][t, i]);
                                    }
                                }
                            }
                            logB[i, m] = lnnum - lndenom;
                        }
                    }
                }
            } while (should_continue);

            return newLogLikelihood;
        }

        /// <summary>Training stops when the improvement is at most <see cref="Tolerance"/> or <see cref="Iterations"/> is reached.</summary>
        /// <param name="change">The improvement of the log-likelihood in the last iteration.</param>
        /// <param name="iteration">The number of iterations done so far.</param>
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
}
