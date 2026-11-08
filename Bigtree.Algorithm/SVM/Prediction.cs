/*
 * SVM.NET Library
 * Copyright (C) 2008 Matthew Johnson
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


using System;
using System.Linq;
using System.IO;
using System.Diagnostics;
using Bigtree.Algorithm.Internal;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace Bigtree.Algorithm.SVM
{
    /// <summary>
    /// Class containing the routines to perform class membership prediction using a trained SVM.
    /// </summary>
    public static class Prediction
    {
        /// <summary>
        /// Predicts the class memberships of all the vectors in the problem.
        /// </summary>
        /// <param name="problem">The SVM Problem to solve</param>
        /// <param name="outputFile">File for result output</param>
        /// <param name="model">The Model to use</param>
        /// <param name="predict_probability">Whether to output a distribution over the classes</param>
        /// <returns>
        ///     For classification, the fraction of samples whose predicted label equals the problem's label; for
        ///     regression (EPSILON_SVR, NU_SVR), the mean squared error.
        /// </returns>
        public static double Predict(
            Problem problem,
            string outputFile,
            Model model,
            bool predict_probability)
        {
            int correct = 0;
            int total = 0;
            double error = 0;
            double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
            StreamWriter output = outputFile != null ? new StreamWriter(outputFile) : null;

            SvmType svm_type = Procedures.svm_get_svm_type(model);
            int nr_class = Procedures.svm_get_nr_class(model);
            int[] labels = new int[nr_class];
            double[] prob_estimates = null;

            if (predict_probability)
            {
                if (svm_type == SvmType.EPSILON_SVR || svm_type == SvmType.NU_SVR)
                {
                    Console.WriteLine("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + Procedures.svm_get_svr_probability(model));
                }
                else
                {
                    Procedures.svm_get_labels(model, labels);
                    prob_estimates = new double[nr_class];
                    if (output != null)
                    {
                        output.Write("labels");
                        for (int j = 0; j < nr_class; j++)
                        {
                            output.Write(" " + labels[j]);
                        }
                        output.Write("\n");
                    }
                }
            }
            for (int i = 0; i < problem.Count; i++)
            {
                double target = problem.Y[i];
                Node[] x = problem.X[i];

                double v;
                if (predict_probability && (svm_type == SvmType.C_SVC || svm_type == SvmType.NU_SVC))
                {
                    v = Procedures.svm_predict_probability(model, x, prob_estimates);
                    if (output != null)
                    {
                        output.Write(v + " ");
                        for (int j = 0; j < nr_class; j++)
                        {
                            output.Write(prob_estimates[j] + " ");
                        }
                        output.Write("\n");
                    }
                }
                else
                {
                    v = Procedures.svm_predict(model, x);
                    if(output != null)
                        output.Write(v + "\n");
                }

                if (v == target)
                    ++correct;
                error += (v - target) * (v - target);
                sumv += v;
                sumy += target;
                sumvv += v * v;
                sumyy += target * target;
                sumvy += v * target;
                ++total;
            }
            if(output != null)
                output.Close();

            if (model.Parameter.SvmType == SvmType.EPSILON_SVR || model.Parameter.SvmType == SvmType.NU_SVR)
                return error / total;
            else return (double)correct / total;
        }

        /// <summary>
        /// Predict the labels for all the data points in a problem.
        /// </summary>
        /// <param name="model">The model to use</param>
        /// <param name="problem">The problem to solve</param>
        /// <returns>The predicted labels</returns>
        public static double[] PredictLabels(this Model model, Problem problem)
        {
            return problem.X.Select(o => model.Predict(o)).ToArray();
        }

        /// <summary>
        /// Predict the probability distributions over all labels for each data point in a problem.
        /// </summary>
        /// <param name="model">The model to use</param>
        /// <param name="problem">The problem to solve</param>
        /// <returns>A distribution over labels for each data point</returns>
        public static double[][] PredictLabelsProbability(this Model model, Problem problem)
        {
            return problem.X.Select(o => model.PredictProbability(o)).ToArray();
        }

        /// <summary>
        ///     libsvm's decision values (<c>svm_predict_values</c>) for every row of a dense sample matrix, computed for
        ///     the whole batch at once on NumSharp.
        /// </summary>
        /// <param name="model">A model trained on 1-based features (<see cref="Problem.FromNDArray"/>, or libsvm files).</param>
        /// <param name="X">
        ///     The samples, shape (samples, features); column <c>j</c> is feature <c>j + 1</c>, and features the model
        ///     knows beyond the last column count as 0.
        /// </param>
        /// <returns>
        ///     Classification: float64 (samples, k(k-1)/2), one column per class pair (i, j) with i &lt; j in
        ///     <see cref="Model.ClassLabels"/> order, positive meaning a vote for class i (libsvm's sign, which for two
        ///     classes is the opposite of scikit-learn's <c>decision_function</c>). Regression and one-class: float64
        ///     (samples,), the regression value or the signed distance. The caller owns it.
        /// </returns>
        /// <remarks>
        ///     <para>The kernel between every sample and every support vector is one matrix: <c>X @ SV.T</c>
        ///     (then <c>(gamma * dot + coef0)^degree</c>, <c>tanh(gamma * dot + coef0)</c>, or
        ///     <c>exp(-gamma * (|x|^2 + |sv|^2 - 2 dot))</c>), and each class pair's values are two matrix-vector
        ///     products over its support vectors. The values agree with the per-row <see cref="Predict(Model, Node[])"/>
        ///     path to rounding (about 1e-15 relative): the sums run in another order.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="model"/> or <paramref name="X"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array.</exception>
        /// <exception cref="NotSupportedException">The model uses a precomputed kernel, which has no feature matrix to compute from.</exception>
        /// <exception cref="InvalidOperationException">A support vector has feature index 0, i.e. the model was trained on 0-based features; use the <see cref="Node"/> API for it.</exception>
        public static NDArray DecisionValues(this Model model, NDArray X)
        {
            if (model is null)
                throw new ArgumentNullException(nameof(model));
            using var scope = NDScope.Open();
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            NDArray supportVectors = DenseSupportVectors(model, (int)samples.shape[1]);
            if (supportVectors.shape[1] > samples.shape[1])
                samples = np.concatenate(new[] { samples, np.zeros(new Shape(samples.shape[0], supportVectors.shape[1] - samples.shape[1])) }, axis: 1);
            NDArray vectorNorms = model.Parameter.KernelType == KernelType.RBF ? np.sum(np.square(supportVectors), axis: 1) : null;

            SvmType type = model.Parameter.SvmType;
            bool single = type == SvmType.ONE_CLASS || type == SvmType.EPSILON_SVR || type == SvmType.NU_SVR;
            NDArray weights, offsets;
            if (single)
            {
                weights = np.array(model.SupportVectorCoefficients[0]);
                offsets = np.array(new[] { model.Rho[0] }).reshape(new Shape());
            }
            else
            {
                (weights, offsets) = PairWeights(model);
            }

            // A block of rows at a time, each in its own scope: the (block, support vectors) kernel dies before the next
            // block allocates its own, so memory stays at one block whatever the batch size (20,000 rows x 1,960
            // support vectors computed at once peaked at 695 MB).
            long rows = samples.shape[0];
            long blockRows = Math.Max(1, BlockElements / Math.Max(1, model.SupportVectorCount));
            NDArray result = single ? np.empty(new Shape(rows)) : np.empty(new Shape(rows, offsets.size));
            for (long start = 0; start < rows; start += blockRows)
            {
                using var block = NDScope.Open();
                string range = $"{start}:{Math.Min(start + blockRows, rows)}";
                NDArray kernel = KernelBlock(model.Parameter, samples[range], supportVectors, vectorNorms);
                result[range] = np.dot(kernel, weights) - offsets;
            }
            return scope.Returns(result);
        }

        /// <summary>
        ///     The number of elements a block's kernel matrix may hold: 4,194,304 float64 (32 MB). Batch prediction computes
        ///     the kernel of that many (sample, support vector) pairs at a time.
        /// </summary>
        private const long BlockElements = 1L << 22;

        /// <summary>
        ///     Every class pair's weights as one (support vectors, pairs) matrix, zero outside the pair's two classes, so
        ///     all pairs come from ONE contiguous matrix product (a BLAS gemm once NumSharp's OpenBLAS package is enabled);
        ///     slicing the kernel per pair instead gave strided operands, which NumSharp's dot computes far slower.
        /// </summary>
        /// <param name="model">A classification model.</param>
        /// <returns>The weights, and every pair's rho to subtract (pairs,).</returns>
        [NDScopedCovered]
        private static (NDArray weights, NDArray offsets) PairWeights(Model model)
        {
            // Support vectors are stored grouped by class: class c owns rows start[c] .. start[c] + count[c].
            int classes = model.NumberOfClasses;
            var start = new int[classes];
            for (int c = 1; c < classes; c++)
                start[c] = start[c - 1] + model.NumberOfSVPerClass[c - 1];

            int pairs = classes * (classes - 1) / 2;
            var weights = new double[(long)model.SupportVectorCount * pairs];
            int p = 0;
            for (int i = 0; i < classes; i++)
                for (int j = i + 1; j < classes; j++)
                {
                    // The same coefficient rows as svm_predict_values: class i's support vectors are weighted by
                    // row j - 1, class j's by row i.
                    for (int k = start[i]; k < start[i] + model.NumberOfSVPerClass[i]; k++)
                        weights[(long)k * pairs + p] = model.SupportVectorCoefficients[j - 1][k];
                    for (int k = start[j]; k < start[j] + model.NumberOfSVPerClass[j]; k++)
                        weights[(long)k * pairs + p] = model.SupportVectorCoefficients[i][k];
                    p++;
                }

            return (np.array(weights).reshape(model.SupportVectorCount, pairs), np.array(model.Rho.Take(pairs).ToArray()));
        }

        /// <summary>
        ///     Predicts every row of a dense sample matrix: the voted class label for classification, the value for
        ///     regression, +1 / -1 for one-class. The batch counterpart of <see cref="Predict(Model, Node[])"/>.
        /// </summary>
        /// <param name="model">A model trained on 1-based features (<see cref="Problem.FromNDArray"/>, or libsvm files).</param>
        /// <param name="X">The samples, shape (samples, features); column <c>j</c> is feature <c>j + 1</c>.</param>
        /// <returns>A float64 vector, one prediction per sample; the caller owns it.</returns>
        /// <remarks>
        ///     Classification votes once per class pair from <see cref="DecisionValues(Model, NDArray)"/> and takes the
        ///     class with the most votes; a tie goes to the class first in <see cref="Model.ClassLabels"/>, as in libsvm.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="model"/> or <paramref name="X"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array.</exception>
        /// <exception cref="NotSupportedException">The model uses a precomputed kernel.</exception>
        /// <exception cref="InvalidOperationException">The model was trained on 0-based features.</exception>
        public static NDArray PredictLabels(this Model model, NDArray X)
        {
            using var scope = NDScope.Open();
            NDArray decision = DecisionValues(model, X);
            SvmType type = model.Parameter.SvmType;
            if (type == SvmType.EPSILON_SVR || type == SvmType.NU_SVR)
                return scope.Returns(decision);
            if (type == SvmType.ONE_CLASS)
                return scope.Returns(np.where(decision > 0, 1.0, -1.0));

            // dec > 0 votes for the pair's first class i, anything else (0 included) for j: svm_predict_values' rule.
            // With firsts[p, c] = (c == i), seconds[p, c] = (c == j), a sample's votes are
            //   positive @ firsts + (1 - positive) @ seconds = positive @ (firsts - seconds) + column sums of seconds,
            // one small matrix product for the whole batch instead of strided per-pair column updates.
            int classes = model.NumberOfClasses;
            int pairs = classes * (classes - 1) / 2;
            var swing = new double[pairs * classes];
            var baseline = new double[classes];
            int p = 0;
            for (int i = 0; i < classes; i++)
                for (int j = i + 1; j < classes; j++)
                {
                    swing[p * classes + i] = 1;
                    swing[p * classes + j] = -1;
                    baseline[j] += 1;
                    p++;
                }

            NDArray positive = (decision > 0).astype(np.float64);
            NDArray votes = np.dot(positive, np.array(swing).reshape(pairs, classes)) + np.array(baseline);

            // np.argmax takes the first maximum: the vote tie rule of libsvm.
            NDArray labels = np.array(model.ClassLabels.Select(label => (double)label).ToArray());
            return scope.Returns(labels[np.argmax(votes, axis: 1)]);
        }

        /// <summary>
        ///     The class probability estimates of every row of a dense sample matrix (a model trained with
        ///     <see cref="Parameter.Probability"/>). The batch counterpart of <see cref="PredictProbability(Model, Node[])"/>.
        /// </summary>
        /// <param name="model">A probability-trained C_SVC or NU_SVC model on 1-based features.</param>
        /// <param name="X">The samples, shape (samples, features); column <c>j</c> is feature <c>j + 1</c>.</param>
        /// <returns>A float64 (samples, classes) array, columns in <see cref="Model.ClassLabels"/> order; the caller owns it.</returns>
        /// <remarks>
        ///     libsvm's pairwise coupling iterates per sample, so this converts each row to <see cref="Node"/>s and runs
        ///     the scalar path; only the input and the result are NumSharp arrays.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="model"/> or <paramref name="X"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array.</exception>
        /// <exception cref="Exception">The model is not a classification model (the message names its type).</exception>
        public static NDArray PredictLabelsProbability(this Model model, NDArray X)
        {
            if (model is null)
                throw new ArgumentNullException(nameof(model));
            using var scope = NDScope.Open();
            Problem rows = Problem.FromNDArray(X, np.zeros(new Shape(X?.shape[0] ?? 0)));
            double[][] probabilities = PredictLabelsProbability(model, rows);
            int classes = model.NumberOfClasses;
            var flat = new double[(long)rows.Count * classes];
            for (int i = 0; i < rows.Count; i++)
                Array.Copy(probabilities[i], 0, flat, (long)i * classes, classes);
            return scope.Returns(np.array(flat).reshape(rows.Count, classes));
        }

        /// <summary>
        ///     The support vectors as a dense matrix: row k, column <c>index - 1</c>.
        /// </summary>
        /// <param name="model">The trained model.</param>
        /// <param name="columns">The query matrix's column count; the result is at least that wide.</param>
        /// <returns>
        ///     A float64 (support vectors, width) array, width = the larger of <paramref name="columns"/> and the highest
        ///     feature index, so a feature the query lacks still counts in the RBF norms (the query is padded with zeros).
        /// </returns>
        /// <exception cref="NotSupportedException">The model uses a precomputed kernel.</exception>
        /// <exception cref="InvalidOperationException">A support vector has feature index 0.</exception>
        [NDScopedCovered]
        private static NDArray DenseSupportVectors(Model model, int columns)
        {
            if (model.Parameter.KernelType == KernelType.PRECOMPUTED)
                throw new NotSupportedException("A precomputed-kernel model has no feature matrix to compute kernels from; use the Node[] API.");

            int vectors = model.SupportVectorCount;
            int width = columns;
            for (int k = 0; k < vectors; k++)
                foreach (Node node in model.SupportVectors[k])
                {
                    if (node.Index < 1)
                        throw new InvalidOperationException("The model has feature index 0 (it was trained on 0-based features): predict with the Node[] API.");
                    width = Math.Max(width, node.Index);
                }

            var dense = new double[(long)vectors * width];
            for (int k = 0; k < vectors; k++)
                foreach (Node node in model.SupportVectors[k])
                    dense[(long)k * width + node.Index - 1] = node.Value;
            return np.array(dense).reshape(vectors, width);
        }

        /// <summary>
        ///     The kernel value of every sample of a block against every support vector, as one NumSharp matrix.
        /// </summary>
        /// <param name="parameter">The model's kernel parameters.</param>
        /// <param name="samples">One block of validated float64 samples, as wide as <paramref name="supportVectors"/>.</param>
        /// <param name="supportVectors">The dense support vectors (<see cref="DenseSupportVectors"/>).</param>
        /// <param name="vectorNorms">The squared norm of every support vector (RBF only, else <c>null</c>).</param>
        /// <returns>A float64 (samples, support vectors) array.</returns>
        /// <exception cref="NotSupportedException">The kernel type is not one of LINEAR, POLY, RBF, SIGMOID.</exception>
        [NDScopedCovered]
        private static NDArray KernelBlock(Parameter parameter, NDArray samples, NDArray supportVectors, NDArray vectorNorms)
        {
            NDArray dot = np.dot(samples, supportVectors.T);
            // The elementwise part of each kernel is one fused np.evaluate pass over the (samples, support vectors)
            // matrix: unfused, the RBF formula alone made six full passes and five matrix-sized temporaries, which on
            // large batches cost more than the matrix product itself. Same operations per element, same results.
            switch (parameter.KernelType)
            {
                case KernelType.LINEAR:
                    return dot;
                case KernelType.POLY:
                    return np.evaluate(NDExpr.Power(parameter.Gamma * (NDExpr)dot + parameter.Coefficient0, (double)parameter.Degree));
                case KernelType.RBF:
                    {
                        NDArray sampleNorms = np.sum(np.square(samples), axis: 1, keepdims: true);       // (samples, 1)
                        // |x - v|^2 = |x|^2 - 2 x.v + |v|^2, which can dip a hair below 0 where a sample equals a
                        // support vector: clamp before the exponential.
                        NDExpr squared = NDExpr.Maximum((NDExpr)sampleNorms - 2.0 * (NDExpr)dot + vectorNorms, 0.0);
                        return np.evaluate(NDExpr.Exp(-parameter.Gamma * squared));
                    }
                case KernelType.SIGMOID:
                    return np.evaluate(NDExpr.Tanh(parameter.Gamma * (NDExpr)dot + parameter.Coefficient0));
                default:
                    throw new NotSupportedException($"Kernel {parameter.KernelType} is not supported.");
            }
        }

        /// <summary>
        /// Predict the class for a single input vector.
        /// </summary>
        /// <param name="model">The Model to use for prediction</param>
        /// <param name="x">The vector for which to predict class</param>
        /// <returns>The result</returns>
        public static double Predict(this Model model, Node[] x)
        {
            return Procedures.svm_predict(model, x);
        }

        /// <summary>
        /// Predicts a class distribution for the single input vector.
        /// </summary>
        /// <param name="model">Model to use for prediction</param>
        /// <param name="x">The vector for which to predict the class distribution</param>
        /// <returns>A probability distribtion over classes</returns>
        public static double[] PredictProbability(this Model model, Node[] x)
        {
            SvmType svm_type = Procedures.svm_get_svm_type(model);
            if (svm_type != SvmType.C_SVC && svm_type != SvmType.NU_SVC)
                throw new Exception("Model type " + svm_type + " unable to predict probabilities.");
            int nr_class = Procedures.svm_get_nr_class(model);
            double[] probEstimates = new double[nr_class];
            Procedures.svm_predict_probability(model, x, probEstimates);
            return probEstimates;
        }

        private static void exit_with_help()
        {
            Debug.Write("usage: svm_predict [options] test_file model_file output_file\n" + "options:\n" + "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n");
            Environment.Exit(1);
        }

        /// <summary>
        /// Legacy method, provided to allow usage as though this were the command line version of libsvm.
        /// </summary>
        /// <param name="args">Standard arguments passed to the svm_predict exectutable.  See libsvm documentation for details.</param>
        [Obsolete("Use the other version of Predict() instead")]
        public static void Predict(params string[] args)
        {
            int i = 0;
            bool predictProbability = false;

            // parse options
            for (i = 0; i < args.Length; i++)
            {
                if (args[i][0] != '-')
                    break;
                ++i;
                switch (args[i - 1][1])
                {

                    case 'b':
                        predictProbability = int.Parse(args[i]) == 1;
                        break;

                    default:
                        throw new ArgumentException("Unknown option");

                }
            }
            if (i >= args.Length)
                throw new ArgumentException("No input, model and output files provided");

            Problem problem = Problem.Read(args[i]);
            Model model = Model.Read(args[i + 1]);
            Predict(problem, args[i + 2], model, predictProbability);
        }
    }
}