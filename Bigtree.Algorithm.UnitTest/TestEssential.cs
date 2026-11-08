using System;
using System.IO;

namespace Bigtree.Algorithm.UnitTest
{
    /// <summary>
    ///     Base class of the tests that read files: it locates the <c>data</c> folder the build copies next to the
    ///     test assembly.
    /// </summary>
    /// <remarks>
    ///     Bigtree 0.4.0 read the folder from a <c>Settings/*.json</c> file under the repository root that was never
    ///     committed, so every test deriving from this class failed before it ran. The data now ships with the test
    ///     project (<c>Bigtree.Algorithm.UnitTest/data</c>, plus the root <c>data/seeds_dataset.csv</c>).
    /// </remarks>
    public abstract class TestEssential
    {
        /// <summary>The <c>data</c> folder beside the test assembly.</summary>
        protected static string DataDir { get; } = Path.Combine(AppContext.BaseDirectory, "data");
    }
}
