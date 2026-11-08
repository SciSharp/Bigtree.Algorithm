using Bigtree.Algorithm.HiddenMarkovModel.Helpers;
using Bigtree.Algorithm.Internal;
using NumSharp;
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
    ///     <para><b>How it computes.</b> The forward and backward recursions run per sequence as scalar loops (each
    ///     time step depends on the previous one and touches only N states, where a NumSharp call would cost more than
    ///     the arithmetic). Everything after them runs on NumSharp over all sequences at once, stacked row after row:
    ///     the state posteriors <c>gamma</c> (rows, N), the transition posteriors <c>ksi</c> (transitions, N, N), and
    ///     the re-estimation, whose emission counts are one matrix product with the one-hot observations. The
    ///     probabilities stay in log space throughout, as before.</para>
    ///     <para><b>Changed in 0.5.0.</b> The convergence log-likelihood sums the likelihood of every sequence (in log
    ///     space) before dividing by the sequence count, as in the Accord.NET algorithm this class derives from;
    ///     Bigtree 0.4.0 reset it for every sequence and so measured only the last one. The discrete
    ///     <see cref="Run(int[][], double[])"/> no longer fills the per-sequence <see cref="mLogGamma"/> and
    ///     <see cref="mLogKsi"/> scratch arrays (the continuous variant still does).</para>
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

            int N = mModel.StateCount;
            double[,] logA = mModel.LogTransitionMatrix;
            double[,] logB = mModel.LogEmissionMatrix;
            double[] logPi = mModel.LogProbabilityVector;

            // The sequences stacked row after row: sequence k occupies rows offset[k] .. offset[k + 1] - 1.
            var offset = new int[K + 1];
            for (int k = 0; k < K; ++k)
                offset[k + 1] = offset[k] + observations_db[k].Length;
            int rows = offset[K];
            int maxT = observations_db.Max(x => x.Length);
            double[,] lnfwd = new double[maxT, N];
            double[,] lnbwd = new double[maxT, N];
            var forward = new double[(long)rows * N];
            var backward = new double[(long)rows * N];

            // The arrays that do not change between iterations live for the whole run in this scope; each iteration
            // opens its own (Iterate is [NDScoped]), so its temporaries return to the pool before the next one.
            using var scope = NDScope.Open();
            var layout = Layout.Create(observations_db, offset, mLogWeights, mModel.SymbolCount);

            // Initialize the model log-likelihoods
            double newLogLikelihood = Double.NegativeInfinity;
            double oldLogLikelihood = Double.NegativeInfinity;
            int iteration = 0;
            bool should_continue = true;

            do // Until convergence or max iterations is reached
            {
                oldLogLikelihood = newLogLikelihood;

                // The recursions stay per sequence and scalar: every step depends on the previous one.
                for (int k = 0; k < K; ++k)
                {
                    int[] observations = observations_db[k];
                    ForwardBackwardAlgorithm.LogForward(logA, logB, logPi, observations, lnfwd);
                    ForwardBackwardAlgorithm.LogBackward(logA, logB, logPi, observations, lnbwd);
                    // Rows 0 .. T-1 of the (maxT, N) buffers are contiguous: one block copy per sequence.
                    int bytes = observations.Length * N * sizeof(double);
                    Buffer.BlockCopy(lnfwd, 0, forward, offset[k] * N * sizeof(double), bytes);
                    Buffer.BlockCopy(lnbwd, 0, backward, offset[k] * N * sizeof(double), bytes);
                }

                iteration++;
                should_continue = Iterate(layout, forward, backward, oldLogLikelihood, iteration, out newLogLikelihood);
            } while (should_continue);

            return newLogLikelihood;
        }

        /// <summary>
        ///     One expectation step over every sequence at once and, unless training has converged, one maximization
        ///     step that writes the new probabilities into the model's arrays.
        /// </summary>
        /// <param name="layout">The stacked observations and their index arrays.</param>
        /// <param name="forward">Every sequence's log forward variables, stacked (rows, N), row-major.</param>
        /// <param name="backward">Every sequence's log backward variables, stacked (rows, N), row-major.</param>
        /// <param name="oldLogLikelihood">The previous iteration's log-likelihood.</param>
        /// <param name="iteration">The 1-based number of this iteration.</param>
        /// <param name="newLogLikelihood">This iteration's log-likelihood, computed before the update.</param>
        /// <returns><c>true</c> when the model was updated and training continues; <c>false</c> when it has converged.</returns>
        [NDScoped]
        private bool Iterate(Layout layout, double[] forward, double[] backward, double oldLogLikelihood, int iteration, out double newLogLikelihood)
        {
            int N = mModel.StateCount;
            NDArray F = np.array(forward).reshape(layout.Rows, N);
            NDArray B = np.array(backward).reshape(layout.Rows, N);
            NDArray logA = FromMatrix(mModel.LogTransitionMatrix);
            NDArray logB = FromMatrix(mModel.LogEmissionMatrix);

            // gamma[t, i] = log P(state i at t | sequence), normalized per row; a row with no mass stays as it is.
            NDArray G = F + B + layout.RowWeight;
            NDArray rowMass = NDHelpers.LogSumExp(G, 1);
            NDArray logGamma = G - np.where(rowMass == double.NegativeInfinity, 0.0, rowMass).reshape(-1, 1);

            // ksi[t, i, j] = log P(i at t, j at t + 1 | sequence) over every transition inside a sequence:
            // forward[t, i] + w + logA[i, j] + backward[t + 1, j] + logB[j, symbol at t + 1], normalized per t.
            NDArray logKsi = null;
            long transitions = layout.Current.size;
            if (transitions > 0)
            {
                NDArray emitted = logB.T[layout.NextSymbol];                        // (transitions, N): logB[j, o(t+1)]
                NDArray from = (F[layout.Current] + layout.RowWeight[layout.Current]).reshape(transitions, N, 1);
                NDArray to = (B[layout.Next] + emitted).reshape(transitions, 1, N);
                NDArray ksi = from + logA.reshape(1, N, N) + to;
                NDArray ksiMass = NDHelpers.LogSumExp(ksi.reshape(transitions, N * N), 1);
                logKsi = ksi - ksiMass.reshape(transitions, 1, 1);
            }

            // log of the summed likelihood of every sequence, over the sequence count (Accord.NET's measure).
            newLogLikelihood = NDHelpers.LogSumExp(F[layout.Last].reshape(1, -1), 1).GetDouble(0) / layout.Sequences;
            if (ShouldTerminate(newLogLikelihood - oldLogLikelihood, iteration))
                return false;

            // pi: the posterior of each state at the first step, averaged over the sequences.
            NDArray logPi = NDHelpers.LogSumExp(logGamma[layout.First], 0) - Math.Log(layout.Sequences);

            // A[i, j] = expected i -> j transitions / expected visits to i before a sequence's last step. 0.4.0 wrote
            // 0 (probability 1) where both are equal, including a never-visited state's -inf / -inf; kept as it was.
            NDArray newLogA;
            if (logKsi is not null)
            {
                NDArray numerator = NDHelpers.LogSumExp(logKsi.reshape(transitions, N * N), 0).reshape(N, N);
                NDArray denominator = NDHelpers.LogSumExp(logGamma[layout.Current], 0).reshape(N, 1);
                newLogA = np.where(numerator == denominator, 0.0, numerator - denominator);
            }
            else
            {
                newLogA = np.zeros(new Shape(N, N));
            }

            // B[i, m] = expected visits to i while emitting m / expected visits to i. The numerator is the posterior
            // mass per symbol: one matrix product with the one-hot observations, shifted by each state's maximum so
            // the exponentials cannot underflow to 0 where the log-space sum would not.
            NDArray peak = np.amax(logGamma, axis: 0);
            NDArray shift = np.where(np.isfinite(peak), peak, 0.0);
            NDArray emissions = np.log(np.dot(np.exp(logGamma - shift).T, layout.Symbols)) + shift.reshape(N, 1);
            NDArray newLogB = emissions - NDHelpers.LogSumExp(logGamma, 0).reshape(N, 1);

            // In place: the model hands out these very arrays through its Log* properties.
            CopyInto(logPi, mModel.LogProbabilityVector);
            CopyInto(newLogA, mModel.LogTransitionMatrix);
            CopyInto(newLogB, mModel.LogEmissionMatrix);
            return true;
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

        /// <summary>A copy of a 2-D managed array as a NumSharp array of the same shape.</summary>
        /// <param name="matrix">The array to copy.</param>
        /// <returns>A float64 array of <paramref name="matrix"/>'s shape; the caller's scope owns it.</returns>
        [NDScopedCovered]
        private static NDArray FromMatrix(double[,] matrix)
        {
            var flat = new double[matrix.Length];
            Buffer.BlockCopy(matrix, 0, flat, 0, flat.Length * sizeof(double));
            return np.array(flat).reshape(matrix.GetLength(0), matrix.GetLength(1));
        }

        /// <summary>Writes a NumSharp array's elements, in row-major order, into an existing managed array of the same size.</summary>
        /// <param name="source">The values.</param>
        /// <param name="target">A <c>double[]</c> or <c>double[,]</c> with as many elements; overwritten in place.</param>
        private static void CopyInto(NDArray source, Array target)
        {
            double[] values = source.ToArray<double>();
            Buffer.BlockCopy(values, 0, target, 0, values.Length * sizeof(double));
        }

        /// <summary>
        ///     The stacked view of a run's observation sequences: what every iteration indexes with. Its arrays are
        ///     created once per <see cref="Run(int[][], double[])"/> under that call's scope, which owns and reclaims
        ///     them (hence borrowed: the layout never disposes them and must not outlive the run).
        /// </summary>
        [NDBorrowed]
        private sealed class Layout
        {
            /// <summary>The total number of observations (rows).</summary>
            public int Rows;

            /// <summary>The number of sequences.</summary>
            public int Sequences;

            /// <summary>The row of every sequence's first observation, int64 (K,).</summary>
            public NDArray First;

            /// <summary>The row of every sequence's last observation, int64 (K,).</summary>
            public NDArray Last;

            /// <summary>Every row that has a successor in the same sequence (the transitions), int64.</summary>
            public NDArray Current;

            /// <summary>The successor of every row of <see cref="Current"/>, int64.</summary>
            public NDArray Next;

            /// <summary>The symbol observed at every row of <see cref="Next"/>, int64.</summary>
            public NDArray NextSymbol;

            /// <summary>The log weight of every row's sequence, float64 (rows, 1).</summary>
            public NDArray RowWeight;

            /// <summary>The one-hot observations, float64 (rows, symbols).</summary>
            public NDArray Symbols;

            /// <summary>Builds the layout of a run.</summary>
            /// <param name="sequences">The observation sequences.</param>
            /// <param name="offset">The first row of every sequence, plus the total row count at the end.</param>
            /// <param name="logWeights">The log weight of every sequence.</param>
            /// <param name="symbols">The model's symbol count.</param>
            /// <returns>The layout; its arrays belong to the caller's scope.</returns>
            [NDScopedCovered]
            public static Layout Create(int[][] sequences, int[] offset, double[] logWeights, int symbols)
            {
                int K = sequences.Length, rows = offset[K];
                var observed = new long[rows];
                var weight = new double[rows];
                var current = new long[rows - K];
                int c = 0;
                for (int k = 0; k < K; ++k)
                {
                    for (int t = 0; t < sequences[k].Length; ++t)
                    {
                        observed[offset[k] + t] = sequences[k][t];
                        weight[offset[k] + t] = logWeights[k];
                        if (t + 1 < sequences[k].Length)
                            current[c++] = offset[k] + t;
                    }
                }

                NDArray observedSymbols = np.array(observed);
                NDArray currentRows = np.array(current);
                NDArray nextRows = currentRows + 1;
                return new Layout
                {
                    Rows = rows,
                    Sequences = K,
                    First = np.array(offset.Take(K).Select(row => (long)row).ToArray()),
                    Last = np.array(offset.Skip(1).Select(row => (long)row - 1).ToArray()),
                    Current = currentRows,
                    Next = nextRows,
                    NextSymbol = observedSymbols[nextRows],
                    RowWeight = np.array(weight).reshape(rows, 1),
                    Symbols = NDHelpers.OneHot(observedSymbols, symbols),
                };
            }
        }
    }
}
