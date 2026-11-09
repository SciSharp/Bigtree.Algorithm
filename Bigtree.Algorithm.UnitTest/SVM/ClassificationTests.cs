using System;
using Bigtree.Algorithm.SVM;
using Bigtree.Algorithm.SVMTest;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Bigtree.Algorithm.UnitTest.SVM
{
    /// <summary>
    ///     Two-class and eight-class problems for every SVM type and kernel. The expected test accuracies are
    ///     scikit-learn 1.8.0's (its <c>SVC</c> / <c>NuSVC</c> run libsvm) on the very same scaled problems, which
    ///     Bigtree reproduces exactly, support-vector counts included. 0.4.0 expected 100% for every kernel; the
    ///     tests never ran (RangeTransform crashed on their 1-based features), and the polynomial kernel's true
    ///     accuracy here is 88% / 51% (C_SVC) and 96% / 81% (NU_SVC).
    /// </summary>
    [TestClass]
    public class ClassificationTests
    {
        /// <summary>The kernels in the order of the expectation rows below.</summary>
        private static readonly KernelType[] Kernels = { KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID };

        /// <summary>scikit-learn's test accuracy on the two-class problem: C_SVC, then NU_SVC, per kernel.</summary>
        private static readonly double[][] TwoClassAccuracy =
        {
            new[] { 1.00, 0.88, 1.00, 1.00 },
            new[] { 1.00, 0.96, 1.00, 1.00 },
        };

        /// <summary>scikit-learn's test accuracy on the eight-class problem: C_SVC, then NU_SVC, per kernel.</summary>
        private static readonly double[][] MulticlassAccuracy =
        {
            new[] { 0.90, 0.51, 0.93, 0.96 },
            new[] { 0.96, 0.81, 0.95, 0.96 },
        };

        /// <summary>Every type and kernel on the two-class problem scores what libsvm scores.</summary>
        [TestMethod]
        public void TestTwoClass()
        {
            SvmType[] svmTypes = new SvmType[]{SvmType.C_SVC, SvmType.NU_SVC};

            for (int s = 0; s < svmTypes.Length; s++)
            {
                for (int k = 0; k < Kernels.Length; k++)
                {
                    double score = testTwoClassModel(100, svmTypes[s], Kernels[k]);

                    Assert.AreEqual(TwoClassAccuracy[s][k], score, 1e-9, string.Format("SVM {0} with Kernel {1} did not train correctly", svmTypes[s], Kernels[k]));
                }
            }
        }

        /// <summary>
        ///     Training with probability estimates fits the same decision function (the Platt calibration runs beside
        ///     it), so the accuracies are the same.
        /// </summary>
        [TestMethod]
        public void TestTwoClassProbability()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };

            for (int s = 0; s < svmTypes.Length; s++)
            {
                for (int k = 0; k < Kernels.Length; k++)
                {
                    double score = testTwoClassModel(100, svmTypes[s], Kernels[k], true);

                    Assert.AreEqual(TwoClassAccuracy[s][k], score, 1e-9, string.Format("SVM {0} with Kernel {1} did not train correctly", svmTypes[s], Kernels[k]));
                }
            }
        }

        /// <summary>Every type and kernel on the eight-class problem scores what libsvm scores.</summary>
        [TestMethod]
        public void TestMulticlass()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };

            for (int s = 0; s < svmTypes.Length; s++)
            {
                for (int k = 0; k < Kernels.Length; k++)
                {
                    double score = testMulticlassModel(8, 100, svmTypes[s], Kernels[k]);

                    Assert.AreEqual(MulticlassAccuracy[s][k], score, 1e-9, string.Format("SVM {0} with Kernel {1} did not train correctly", svmTypes[s], Kernels[k]));
                }
            }
        }

        /// <summary>The eight-class problem trained with probability estimates scores the same.</summary>
        [TestMethod]
        public void TestMulticlassProbability()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };

            for (int s = 0; s < svmTypes.Length; s++)
            {
                for (int k = 0; k < Kernels.Length; k++)
                {
                    double score = testMulticlassModel(8, 100, svmTypes[s], Kernels[k], true);

                    Assert.AreEqual(MulticlassAccuracy[s][k], score, 1e-9, string.Format("SVM {0} with Kernel {1} did not train correctly", svmTypes[s], Kernels[k]));
                }
            }
        }

        /// <summary>Trains on the two-class training problem and scores the two-class test problem.</summary>
        /// <param name="count">The number of samples.</param>
        /// <param name="svm">The SVM type.</param>
        /// <param name="kernel">The kernel.</param>
        /// <param name="probability">Whether to train probability estimates.</param>
        /// <param name="outputFile">An optional prediction output file.</param>
        /// <returns>The test accuracy.</returns>
        private double testTwoClassModel(int count, SvmType svm, KernelType kernel, bool probability = false, string outputFile = null)
        {
            Problem train = SVMUtilities.CreateTwoClassProblem(count);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = svm;
            param.KernelType = kernel;
            param.Probability = probability;
            if (svm == SvmType.C_SVC)
            {
                param.Weights[-1] = 1;
                param.Weights[1] = 1;
            }

            Model model = Training.Train(scaled, param);

            Problem test = SVMUtilities.CreateTwoClassProblem(count, false);
            scaled = transform.Scale(test);
            return Prediction.Predict(scaled, outputFile, model, false);
        }

        /// <summary>Trains on the multiclass training problem and scores the multiclass test problem.</summary>
        /// <param name="numberOfClasses">The number of classes.</param>
        /// <param name="count">The number of samples.</param>
        /// <param name="svm">The SVM type.</param>
        /// <param name="kernel">The kernel.</param>
        /// <param name="probability">Whether to train probability estimates.</param>
        /// <param name="outputFile">An optional prediction output file.</param>
        /// <returns>The test accuracy.</returns>
        private double testMulticlassModel(int numberOfClasses, int count, SvmType svm, KernelType kernel, bool probability = false, string outputFile = null)
        {
            Problem train = SVMUtilities.CreateMulticlassProblem(numberOfClasses, count);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = 1.0 / 3;
            param.SvmType = svm;
            param.KernelType = kernel;
            param.Probability = probability;
            if (svm == SvmType.C_SVC)
            {
                for (int i = 0; i < numberOfClasses; i++)
                    param.Weights[i] = 1;
            }

            Model model = Training.Train(scaled, param);

            Problem test = SVMUtilities.CreateMulticlassProblem(numberOfClasses, count, false);
            scaled = transform.Scale(test);
            return Prediction.Predict(scaled, outputFile, model, false);
        }
    }
}
