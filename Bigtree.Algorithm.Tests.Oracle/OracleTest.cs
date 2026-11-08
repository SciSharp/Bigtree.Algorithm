using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Bigtree.Algorithm.Tests.Oracle
{
    /// <summary>
    ///     Base class of every oracle test: a fresh Python namespace per test (with numpy imported), the comparison
    ///     helpers, and a check that the test left no memory shared between NumSharp and Python.
    /// </summary>
    /// <remarks>
    ///     <para><b>Namespace.</b> <see cref="py"/> is a new module scope for each test: <c>py.x = nd</c> binds a NumSharp
    ///     array as a NumPy view, <c>NDArray a = py.x</c> reads one back (the codec), <c>py.Exec("...")</c> runs
    ///     statements. Use it only inside <c>using (Py.GIL())</c>.</para>
    ///     <para><b>Leak check.</b> <see cref="NDArrayPythonInterop.LiveExports"/> / <see cref="NDArrayPythonInterop.LiveImports"/>
    ///     count NumSharp buffers Python still views and Python memory NumSharp still leases; after each test both
    ///     must return to their value before it.</para>
    /// </remarks>
    public abstract class OracleTest
    {
        /// <summary>The running test's Python namespace (see the class remarks).</summary>
        protected dynamic py = null!;

        /// <summary>The module scope behind <see cref="py"/>; <c>null</c> when no engine runs.</summary>
        private PyModule? _namespace;

        /// <summary>The value of <see cref="NDArrayPythonInterop.LiveExports"/> before the test.</summary>
        private int _exportsBefore;

        /// <summary>The value of <see cref="NDArrayPythonInterop.LiveImports"/> before the test.</summary>
        private int _importsBefore;

        /// <summary>Reports Inconclusive without an engine; otherwise records the leak counters and opens the namespace.</summary>
        /// <exception cref="AssertInconclusiveException">No interpreter was configured (<see cref="PythonEnvironment.Skipped"/>).</exception>
        [TestInitialize]
        public void OpenNamespace()
        {
            if (PythonEnvironment.Skipped is string reason)
                Assert.Inconclusive(reason);

            _exportsBefore = NDArrayPythonInterop.LiveExports;
            _importsBefore = NDArrayPythonInterop.LiveImports;
            using (Py.GIL())
            {
                _namespace = Py.CreateScope();
                _namespace.Exec("import numpy as np");
            }
            py = _namespace;
        }

        /// <summary>Drops the namespace, then waits for both leak counters to return to their value before the test.</summary>
        /// <exception cref="AssertFailedException">After 15 seconds of collecting, memory is still shared with Python.</exception>
        [TestCleanup]
        public void CloseNamespaceAndCheckForLeaks()
        {
            if (_namespace is null)
                return;
            using (Py.GIL())
                _namespace.Dispose();
            _namespace = null;

            // A dynamic read leaves a .NET wrapper whose Python reference is released only after the .NET collector
            // finalized it and pythonnet processed its queue: collect until the counters are back.
            var deadline = Stopwatch.StartNew();
            while (!CountersAreBack() && deadline.Elapsed < TimeSpan.FromSeconds(15))
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                using (Py.GIL())
                {
                    Finalizer.Instance.Collect();
                    PythonEngine.RunSimpleString("import gc; gc.collect()");
                    using NDArray probe = np.arange(1);
                    using PyObject conversion = probe.ToNumpyCopy();
                }
            }

            Assert.IsTrue(CountersAreBack(),
                $"The test left memory shared with Python: LiveExports {_exportsBefore} -> {NDArrayPythonInterop.LiveExports}, " +
                $"LiveImports {_importsBefore} -> {NDArrayPythonInterop.LiveImports}.");
        }

        /// <summary>
        ///     An independent NumSharp copy of an oracle value: a NumPy array read from <c>py</c> arrives as a
        ///     <see cref="PyObject"/> (the codec converts on assignment, not when the value is passed as an argument).
        /// </summary>
        /// <param name="oracle">An <see cref="NDArray"/>, or a Python object exposing the buffer protocol.</param>
        /// <returns>A copy the caller disposes; Python keeps no lease on it.</returns>
        /// <exception cref="ArgumentException"><paramref name="oracle"/> is neither.</exception>
        protected static NDArray Copy(object oracle) => oracle switch
        {
            NDArray array => array.copy(),
            PyObject python => UsingGil(() => python.ToNDArray()),
            _ => throw new ArgumentException($"Cannot compare a {oracle?.GetType().Name ?? "null"}.", nameof(oracle)),
        };

        /// <summary>Runs <paramref name="body"/> holding the GIL (re-entrant: a caller that already holds it is fine).</summary>
        /// <typeparam name="T">The result type.</typeparam>
        /// <param name="body">The work.</param>
        /// <returns>The work's result.</returns>
        private static T UsingGil<T>(Func<T> body)
        {
            using (Py.GIL())
                return body();
        }

        /// <summary>Whether both leak counters are at or below their value before the test.</summary>
        /// <returns><c>true</c> when nothing this test shared is still alive.</returns>
        private bool CountersAreBack() =>
            NDArrayPythonInterop.LiveExports <= _exportsBefore && NDArrayPythonInterop.LiveImports <= _importsBefore;

        /// <summary>
        ///     Asserts two arrays have the same shape and the same float64 values to the last bit, and on failure reports
        ///     how many elements differ and the largest difference in units in the last place.
        /// </summary>
        /// <param name="oracle">The oracle's array: an <see cref="NDArray"/>, or a NumPy array as read from <c>py</c> (copied in).</param>
        /// <param name="actual">Bigtree's array.</param>
        /// <param name="what">What is compared, for the message.</param>
        protected static void AssertBitIdentical(object oracle, NDArray actual, string what)
        {
            using NDArray expected = Copy(oracle);
            CollectionAssert.AreEqual(expected.shape, actual.shape, $"{what}: shape");
            double[] e = expected.astype(np.float64).ToArray<double>(), a = actual.astype(np.float64).ToArray<double>();
            int differing = 0;
            long worstUlps = 0;
            for (int i = 0; i < e.Length; i++)
            {
                long ulps = Math.Abs(BitConverter.DoubleToInt64Bits(e[i]) - BitConverter.DoubleToInt64Bits(a[i]));
                if (ulps != 0)
                {
                    differing++;
                    worstUlps = Math.Max(worstUlps, ulps);
                }
            }
            Assert.AreEqual(0, differing, $"{what}: {differing} of {e.Length} elements differ, by up to {worstUlps} ulp.");
        }

        /// <summary>Asserts two arrays have the same shape and values within <c>atol + rtol * |expected|</c>.</summary>
        /// <param name="oracle">The oracle's array: an <see cref="NDArray"/>, or a NumPy array as read from <c>py</c> (copied in).</param>
        /// <param name="actual">Bigtree's array.</param>
        /// <param name="rtol">The relative tolerance.</param>
        /// <param name="atol">The absolute tolerance.</param>
        /// <param name="what">What is compared, for the message.</param>
        protected static void AssertClose(object oracle, NDArray actual, double rtol, double atol, string what)
        {
            using NDArray expected = Copy(oracle);
            CollectionAssert.AreEqual(expected.shape, actual.shape, $"{what}: shape");
            double[] e = expected.astype(np.float64).ToArray<double>(), a = actual.astype(np.float64).ToArray<double>();
            double worst = 0;
            int worstAt = -1;
            for (int i = 0; i < e.Length; i++)
            {
                double excess = Math.Abs(e[i] - a[i]) - (atol + rtol * Math.Abs(e[i]));
                if (excess > worst || double.IsNaN(excess))
                {
                    worst = double.IsNaN(excess) ? double.PositiveInfinity : excess;
                    worstAt = i;
                }
            }
            Assert.AreEqual(-1, worstAt, worstAt < 0 ? "" : $"{what}: element {worstAt} is {a[worstAt]:R}, the oracle says {e[worstAt]:R}.");
        }

        /// <summary>Asserts two arrays are equal element by element (labels, indices, counts).</summary>
        /// <param name="oracle">The oracle's array: an <see cref="NDArray"/>, or a NumPy array as read from <c>py</c> (copied in).</param>
        /// <param name="actual">Bigtree's array.</param>
        /// <param name="what">What is compared, for the message.</param>
        protected static void AssertEqual(object oracle, NDArray actual, string what)
        {
            using NDArray expected = Copy(oracle);
            CollectionAssert.AreEqual(expected.shape, actual.shape, $"{what}: shape");
            CollectionAssert.AreEqual(expected.astype(np.float64).ToArray<double>(), actual.astype(np.float64).ToArray<double>(), what);
        }
    }
}
