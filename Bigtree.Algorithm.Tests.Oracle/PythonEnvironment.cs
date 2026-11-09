using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

// One embedded interpreter serves the whole run, and it has one global lock (the GIL): run the tests one by one,
// or every test would wait for its neighbours' GIL and the leak check of OracleTest would count their conversions.
[assembly: DoNotParallelize]

namespace Bigtree.Algorithm.Tests.Oracle
{
    /// <summary>
    ///     Starts the embedded CPython the oracle compares against, once for the test run, and shuts it down at the end.
    ///     The interpreter is the one named by <see cref="InterpreterVariable"/>; nothing is searched for.
    /// </summary>
    /// <remarks>
    ///     <para>Adapted from NumSharp's <c>examples/NumSharpPythonnetGhActionsCicd</c>, which drives the same
    ///     embedding through <c>NumSharp.Interop.pythonnet</c>.</para>
    ///     <para>Without the variable the run is Inconclusive on a developer machine and fails under <c>CI=true</c>: a
    ///     build server whose oracle silently skips proves nothing.</para>
    /// </remarks>
    [TestClass]
    public sealed class PythonEnvironment
    {
        /// <summary>
        ///     The environment variable naming the Python interpreter to embed; it must have NumPy, SciPy and
        ///     scikit-learn. The same variable NumSharp's pythonnet tests use, so one setting serves both.
        /// </summary>
        public const string InterpreterVariable = "NUMSHARP_PYTHONNET_PYTHON";

        /// <summary>The modules every oracle needs; all must import, or the run fails at once with the module's name.</summary>
        private static readonly string[] Libraries = { "numpy", "scipy", "sklearn" };

        /// <summary>Why no engine runs, or <c>null</c> when one does; each test then reports Inconclusive with this text.</summary>
        public static string? Skipped { get; private set; }

        /// <summary>The versions of Python and the libraries the oracle ran, for the test log.</summary>
        public static string Versions { get; private set; } = "";

        /// <summary>
        ///     Starts the engine inside the named interpreter (a virtual environment included), imports the libraries
        ///     and registers NumSharp's NumPy codec.
        /// </summary>
        /// <param name="context">The MSTest context of the run (unused).</param>
        /// <exception cref="InvalidOperationException">
        ///     The variable is unset under <c>CI=true</c>, the interpreter cannot be embedded, or a library is missing.
        /// </exception>
        [AssemblyInitialize]
        public static void Start(TestContext context)
        {
            string? interpreter = Environment.GetEnvironmentVariable(InterpreterVariable);
            if (string.IsNullOrWhiteSpace(interpreter))
            {
                if (string.Equals(Environment.GetEnvironmentVariable("CI"), "true", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException($"{InterpreterVariable} is not set, so the oracle would compare against nothing.");
                Skipped = $"Set {InterpreterVariable} to a Python interpreter with numpy, scipy and scikit-learn to run the oracle.";
                return;
            }

            PythonInstall install = PythonInstall.Probe(interpreter);
            Runtime.PythonDLL = install.Library;
            // CPython derives sys.prefix and sys.path from the program name; without it a virtual environment is not
            // adopted (its packages are missing) and on Windows sys.executable is the test host.
            PythonEngine.ProgramName = install.Executable;
            PythonEngine.Initialize();
            // Initialize leaves this thread holding the GIL; MSTest runs the tests on other threads.
            PythonEngine.BeginAllowThreads();

            using (Py.GIL())
            {
                var versions = new List<string> { $"Python {install.Version}" };
                foreach (string module in Libraries)
                {
                    try
                    {
                        using PyObject imported = Py.Import(module);
                        using PyObject version = imported.GetAttr("__version__");
                        versions.Add($"{module} {version}");
                    }
                    catch (PythonException e)
                    {
                        throw new InvalidOperationException($"'{install.Executable}' cannot import '{module}' ({e.Message}).", e);
                    }
                }
                Versions = string.Join(", ", versions);
            }

            // NDArray <-> numpy.ndarray for every dynamic call, before the first decode (pythonnet caches a failed lookup).
            NDArrayPythonInterop.RegisterCodec();
        }

        /// <summary>Shuts the engine down after the last test.</summary>
        /// <exception cref="InvalidOperationException">A Python error is still pending on the calling thread.</exception>
        [AssemblyCleanup]
        public static void Stop()
        {
            if (Skipped is not null || !PythonEngine.IsInitialized)
                return;
            // pythonnet saves its state with BinaryFormatter on shutdown, which .NET 8+ removed; NoopFormatter skips it.
            RuntimeData.FormatterType = typeof(NoopFormatter);
            PythonEngine.Shutdown();
        }
    }

    /// <summary>What one interpreter reports about itself: enough to load its shared library and adopt a venv.</summary>
    /// <param name="Executable">The interpreter's own path (<c>sys.executable</c>).</param>
    /// <param name="Version">The Python version.</param>
    /// <param name="Prefix"><c>sys.prefix</c>.</param>
    /// <param name="BasePrefix"><c>sys.base_prefix</c>: the install that owns the shared library.</param>
    /// <param name="Library">The shared library pythonnet loads.</param>
    public sealed record PythonInstall(string Executable, Version Version, string Prefix, string BasePrefix, string Library)
    {
        /// <summary>The program the interpreter runs to describe itself, as one JSON line.</summary>
        private const string Describe =
            "import json, sys, sysconfig\n" +
            "print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable,\n" +
            "                  'prefix': sys.prefix, 'base_prefix': sys.base_prefix,\n" +
            "                  'instsoname': sysconfig.get_config_var('INSTSONAME') or '',\n" +
            "                  'libdir': sysconfig.get_config_var('LIBDIR') or ''}))";

        /// <summary>Runs the named interpreter once and reads what it reports.</summary>
        /// <param name="interpreter">The interpreter's path.</param>
        /// <returns>The install, with a shared library that exists.</returns>
        /// <exception cref="InvalidOperationException">The interpreter does not run, is not CPython, or has no shared library.</exception>
        public static PythonInstall Probe(string interpreter)
        {
            string output, errors;
            int exitCode;
            try
            {
                var start = new ProcessStartInfo(interpreter)
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                start.ArgumentList.Add("-c");
                start.ArgumentList.Add(Describe);
                using Process process = Process.Start(start)!;
                // Read the error stream concurrently: a child filling one pipe while the parent waits on the other blocks both.
                Task<string> pendingErrors = process.StandardError.ReadToEndAsync();
                output = process.StandardOutput.ReadToEnd();
                errors = pendingErrors.GetAwaiter().GetResult();
                process.WaitForExit();
                exitCode = process.ExitCode;
            }
            catch (Win32Exception e)
            {
                throw new InvalidOperationException($"{PythonEnvironment.InterpreterVariable}='{interpreter}' cannot be started: {e.Message}", e);
            }

            string? report = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
            if (exitCode != 0 || report is null)
                throw new InvalidOperationException($"'{interpreter}' did not run (exit code {exitCode}: {errors.Trim()}).");

            using JsonDocument json = JsonDocument.Parse(report);
            JsonElement root = json.RootElement;
            JsonElement v = root.GetProperty("version");
            var version = new Version(v[0].GetInt32(), v[1].GetInt32(), v[2].GetInt32());
            string executable = root.GetProperty("executable").GetString() ?? "";
            string prefix = root.GetProperty("prefix").GetString() ?? "";
            string basePrefix = root.GetProperty("base_prefix").GetString() ?? "";
            string soName = root.GetProperty("instsoname").GetString() ?? "";
            string libDir = root.GetProperty("libdir").GetString() ?? "";

            string? library;
            if (OperatingSystem.IsWindows())
            {
                library = Path.Combine(basePrefix, $"python{version.Major}{version.Minor}.dll");
            }
            else
            {
                string name = $"libpython{version.Major}.{version.Minor}.{(OperatingSystem.IsMacOS() ? "dylib" : "so")}";
                library = new[]
                {
                    soName.Length > 0 && libDir.Length > 0 ? Path.Combine(libDir, soName) : "",
                    soName.Length > 0 ? Path.Combine(basePrefix, Path.GetFileName(soName)) : "",
                    libDir.Length > 0 ? Path.Combine(libDir, name) : "",
                    Path.Combine(basePrefix, "lib", name),
                }.FirstOrDefault(candidate => candidate.Length > 0 && File.Exists(candidate));
            }

            if (library is null || !File.Exists(library))
                throw new InvalidOperationException($"The shared library of '{interpreter}' was not found under '{basePrefix}'.");
            return new PythonInstall(executable, version, prefix, basePrefix, library);
        }
    }
}
