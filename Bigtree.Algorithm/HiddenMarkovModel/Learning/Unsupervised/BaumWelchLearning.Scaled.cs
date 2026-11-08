using Bigtree.Algorithm.HiddenMarkovModel.MathHelpers;
using System;

namespace Bigtree.Algorithm.HiddenMarkovModel.Learning.Unsupervised
{
    public partial class BaumWelchLearning
    {
        /// <summary>
        ///     Baum-Welch with Rabiner's scaled forward-backward: every time step is normalized by its scale
        ///     <c>c_t = P(o_t | o_1..o_t-1)</c>, so the recursions are plain multiply-adds instead of log-space sums (one
        ///     <c>exp</c> and one <c>log1p</c> per term), and the transition statistics of a whole iteration reduce to
        ///     <c>A ∘ (Fᵀ W)</c>: no per-time-step (N, N) posterior is ever stored.
        /// </summary>
        /// <param name="observations_db">The validated observation sequences.</param>
        /// <param name="logLikelihood">The final average log-likelihood, as <see cref="RunScalar"/> reports it.</param>
        /// <returns>
        ///     <c>true</c> when training ran; <c>false</c> (model untouched) when a sequence is impossible under the
        ///     starting model (a scale of 0), which only the log-space loops represent: the caller runs those instead.
        /// </returns>
        /// <remarks>
        ///     <para>Same mathematics as the log-space loops: gamma = α̂·β̂ and ksi = α̂_t[i] A[i, j] B[j, o_t+1]
        ///     β̂_t+1[j] / c_t+1 both sum to 1, and log P(sequence) = Σ log c_t. The results agree with the log-space
        ///     re-estimation to rounding (the unit tests compare with 0.4.0's loops at 1e-9). EM never lowers the
        ///     likelihood, so data possible under the starting model stays possible; only underflow of a single step's
        ///     probability below 1e-308 could still reach a zero scale, and then the run restarts in log space from the
        ///     starting model.</para>
        ///     <para>Sequence weights cancel exactly as in the log-space loops (they are added to every posterior of a
        ///     time step before its normalization), so they are not applied here either.</para>
        /// </remarks>
        private bool RunScaled(int[][] observations_db, out double logLikelihood)
        {
            int K = observations_db.Length, N = mModel.StateCount, M = mModel.SymbolCount;
            double lnK = Math.Log(K);
            double[,] logA = mModel.LogTransitionMatrix, logB = mModel.LogEmissionMatrix;
            double[] logPi = mModel.LogProbabilityVector;
            int maxT = 0;
            foreach (int[] sequence in observations_db)
                maxT = Math.Max(maxT, sequence.Length);

            // Probability-space copies, row-major; emissions transposed so a symbol's column is one contiguous row.
            var A = new double[N * N];
            var Bt = new double[M * N];
            var pi = new double[N];
            // Per-iteration accumulators: Fᵀ W (multiplied by A at the end), the visit sums, the emission sums.
            var outer = new double[N * N];
            var visitsBeforeLast = new double[N];
            var visits = new double[N];
            var emitted = new double[N * M];
            var first = new double[N];
            // Per-sequence buffers, reused.
            var alpha = new double[maxT * N];
            var beta = new double[maxT * N];
            var scale = new double[maxT];
            var w = new double[N];

            double newLogLikelihood = double.NegativeInfinity;
            double oldLogLikelihood;
            int iteration = 0;
            while (true)
            {
                oldLogLikelihood = newLogLikelihood;
                for (int i = 0; i < N; i++)
                {
                    pi[i] = Math.Exp(logPi[i]);
                    for (int j = 0; j < N; j++)
                        A[i * N + j] = Math.Exp(logA[i, j]);
                    for (int m = 0; m < M; m++)
                        Bt[m * N + i] = Math.Exp(logB[i, m]);
                }
                Array.Clear(outer, 0, outer.Length);
                Array.Clear(visitsBeforeLast, 0, N);
                Array.Clear(visits, 0, N);
                Array.Clear(emitted, 0, emitted.Length);
                Array.Clear(first, 0, N);
                double summedLikelihood = double.NegativeInfinity;

                foreach (int[] o in observations_db)
                {
                    int T = o.Length;
                    if (!ScaledForward(o, A, Bt, pi, N, alpha, scale))
                    {
                        logLikelihood = double.NaN;
                        return false;
                    }
                    ScaledBackward(o, A, Bt, N, scale, beta, w);

                    double sequenceLogLikelihood = 0;
                    for (int t = 0; t < T; t++)
                        sequenceLogLikelihood += Math.Log(scale[t]);
                    summedLikelihood = LogHelper.LogSum(summedLikelihood, sequenceLogLikelihood);

                    for (int t = 0; t < T; t++)
                    {
                        int row = t * N, symbol = o[t];
                        for (int i = 0; i < N; i++)
                        {
                            double gamma = alpha[row + i] * beta[row + i];
                            visits[i] += gamma;
                            emitted[i * M + symbol] += gamma;
                            if (t + 1 < T)
                                visitsBeforeLast[i] += gamma;
                            if (t == 0)
                                first[i] += gamma;
                        }
                    }

                    // Σ_t ksi_t = A ∘ Σ_t α̂_t ⊗ w_t with w_t[j] = B[j, o_t+1] β̂_t+1[j] / c_t+1: one outer-product
                    // accumulation per transition, a row update (SAXPY) per state, A applied once after all sequences.
                    for (int t = 0; t + 1 < T; t++)
                    {
                        int next = (t + 1) * N, column = o[t + 1] * N;
                        double c = scale[t + 1];
                        for (int j = 0; j < N; j++)
                            w[j] = Bt[column + j] * beta[next + j] / c;
                        int row = t * N;
                        for (int i = 0; i < N; i++)
                        {
                            double a = alpha[row + i];
                            if (a == 0)
                                continue;
                            int target = i * N;
                            for (int j = 0; j < N; j++)
                                outer[target + j] += a * w[j];
                        }
                    }
                }

                // log(Σ_k P(sequence_k)) / K, the measure the log-space loops compare with the tolerance.
                newLogLikelihood = summedLikelihood / K;
                iteration++;
                if (ShouldTerminate(newLogLikelihood - oldLogLikelihood, iteration))
                    break;

                for (int i = 0; i < N; i++)
                {
                    logPi[i] = Math.Log(first[i]) - lnK;
                    for (int j = 0; j < N; j++)
                    {
                        double numerator = A[i * N + j] * outer[i * N + j];
                        // 0.4.0's rule for equal numerator and denominator (a never-visited state included): log 1.
                        logA[i, j] = numerator == visitsBeforeLast[i] ? 0 : Math.Log(numerator) - Math.Log(visitsBeforeLast[i]);
                    }
                    for (int m = 0; m < M; m++)
                        logB[i, m] = Math.Log(emitted[i * M + m]) - Math.Log(visits[i]);
                }
            }

            logLikelihood = newLogLikelihood;
            return true;
        }

        /// <summary>
        ///     The scaled forward pass of one sequence: α̂_t = α_t / P(o_1..o_t), with the scales
        ///     <c>c_t = P(o_t | o_1..o_t-1)</c> stored beside it.
        /// </summary>
        /// <param name="o">The observations.</param>
        /// <param name="A">The transition matrix, row-major (N × N).</param>
        /// <param name="Bt">The emission matrix transposed, row-major (symbols × N).</param>
        /// <param name="pi">The initial distribution.</param>
        /// <param name="N">The state count.</param>
        /// <param name="alpha">Receives α̂, row-major (T × N).</param>
        /// <param name="scale">Receives c_t.</param>
        /// <returns><c>false</c> as soon as a step has probability 0 (the sequence is impossible under the model).</returns>
        private static bool ScaledForward(int[] o, double[] A, double[] Bt, double[] pi, int N, double[] alpha, double[] scale)
        {
            double c = 0;
            int column = o[0] * N;
            for (int i = 0; i < N; i++)
                c += alpha[i] = pi[i] * Bt[column + i];
            if (!(c > 0))
                return false;
            scale[0] = c;
            for (int i = 0; i < N; i++)
                alpha[i] /= c;

            for (int t = 1; t < o.Length; t++)
            {
                int previous = (t - 1) * N, row = t * N;
                // α_t = (α̂_t-1 A) ∘ B[:, o_t]: accumulate the rows of A weighted by α̂_t-1 (contiguous reads).
                Array.Clear(alpha, row, N);
                for (int j = 0; j < N; j++)
                {
                    double a = alpha[previous + j];
                    if (a == 0)
                        continue;
                    int source = j * N;
                    for (int i = 0; i < N; i++)
                        alpha[row + i] += a * A[source + i];
                }

                column = o[t] * N;
                c = 0;
                for (int i = 0; i < N; i++)
                    c += alpha[row + i] *= Bt[column + i];
                if (!(c > 0))
                    return false;
                scale[t] = c;
                for (int i = 0; i < N; i++)
                    alpha[row + i] /= c;
            }
            return true;
        }

        /// <summary>
        ///     The scaled backward pass of one sequence, with the forward pass's scales: β̂_T-1 = 1 and
        ///     β̂_t[i] = Σ_j A[i, j] B[j, o_t+1] β̂_t+1[j] / c_t+1, so α̂_t[i] β̂_t[i] is the state posterior.
        /// </summary>
        /// <param name="o">The observations.</param>
        /// <param name="A">The transition matrix, row-major (N × N).</param>
        /// <param name="Bt">The emission matrix transposed, row-major (symbols × N).</param>
        /// <param name="N">The state count.</param>
        /// <param name="scale">The forward pass's scales.</param>
        /// <param name="beta">Receives β̂, row-major (T × N).</param>
        /// <param name="u">Scratch of N elements (overwritten).</param>
        private static void ScaledBackward(int[] o, double[] A, double[] Bt, int N, double[] scale, double[] beta, double[] u)
        {
            int T = o.Length;
            int last = (T - 1) * N;
            for (int i = 0; i < N; i++)
                beta[last + i] = 1;

            for (int t = T - 2; t >= 0; t--)
            {
                int next = (t + 1) * N, column = o[t + 1] * N, row = t * N;
                double c = scale[t + 1];
                for (int j = 0; j < N; j++)
                    u[j] = Bt[column + j] * beta[next + j];
                // β̂_t[i] = (row i of A) · u / c: a contiguous dot product per state.
                for (int i = 0; i < N; i++)
                {
                    double sum = 0;
                    int source = i * N;
                    for (int j = 0; j < N; j++)
                        sum += A[source + j] * u[j];
                    beta[row + i] = sum / c;
                }
            }
        }
    }
}
