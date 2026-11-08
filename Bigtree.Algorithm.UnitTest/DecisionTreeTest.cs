using Bigtree.Algorithm.DecisionTree;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest
{
    /// <summary>
    ///     The ID3 tree: the restored NeedUmbrella table, the tree's shape and gains, and the two behaviours 0.5.0
    ///     corrected (multi-class entropy, subtables that keep the columns left of the split).
    /// </summary>
    [TestClass]
    public class DecisionTreeTest : TestEssential
    {
        /// <summary>
        ///     The NeedUmbrella table (Quinlan's 14-day weather table with inverted labels, from WolfgangOfner/DecisionTree,
        ///     where this ID3 implementation came from): rain with weak wind needs no umbrella.
        /// </summary>
        [TestMethod]
        public void TestID3()
        {
            // train
            string filePath = Path.Combine(DataDir, "DecisionTree.csv");
            var data = new DataReader().Read(filePath);

            var decisionTree = new Tree();
            decisionTree.Root = Tree.Learn(data, "");

            // predict
            Dictionary<string, string> testValues = new Dictionary<string, string>();
            testValues.Add("Outlook", "Rain");
            testValues.Add("Temperatur", "Mild");
            testValues.Add("Humidity", "High");
            testValues.Add("Wind", "Weak");

            var result = Tree.CalculateResult(decisionTree.Root, testValues, "");

            Tree.Print(null, result);
            Tree.PrintLegend("The colors indicate the following values:");

            Assert.IsTrue(result.Equals("OUTLOOK -- rain --> WIND -- weak --> NO"));
        }

        /// <summary>
        ///     The whole tree of the NeedUmbrella table: Outlook at the root with its values in order of first
        ///     appearance, Overcast a leaf, Sunny split by Humidity and Rain by Wind. Pins the gains and the
        ///     first-appearance edge order that the NumSharp counting must reproduce.
        /// </summary>
        [TestMethod]
        public void NeedUmbrellaTreeShape()
        {
            var data = new DataReader().Read(Path.Combine(DataDir, "DecisionTree.csv"));
            TreeNode root = Tree.Learn(data, "");

            Assert.AreEqual("Outlook", root.Name);
            CollectionAssert.AreEqual(new[] { "Sunny", "Overcast", "Rain" }, root.NodeAttribute.DifferentAttributeNames);
            // ID3's textbook gains on this table (the labels are inverted, which leaves every entropy unchanged).
            Assert.AreEqual(0.2467498197744391, root.NodeAttribute.InformationGain, 1e-12);

            var sunny = root.ChildNodes.Single(child => child.Edge == "Sunny");
            var overcast = root.ChildNodes.Single(child => child.Edge == "Overcast");
            var rain = root.ChildNodes.Single(child => child.Edge == "Rain");
            Assert.AreEqual("Humidity", sunny.Name);
            Assert.IsTrue(overcast.IsLeaf);
            Assert.AreEqual("No", overcast.Name);
            Assert.AreEqual("Wind", rain.Name);
        }

        /// <summary>
        ///     Three labels: the entropy is the multi-class one. Bigtree 0.4.0 split the labels into "the first row's
        ///     label" and "the rest" for the attributes, which cannot tell B from C here and picked the useless column.
        /// </summary>
        [TestMethod]
        public void MultiClassLabelsUseTheFullEntropy()
        {
            var data = new TrainingData
            {
                Columns = new[] { "Noise", "Signal", "Class" },
                Rows = new[]
                {
                    new[] { "x", "a", "A" },
                    new[] { "y", "a", "A" },
                    new[] { "x", "b", "B" },
                    new[] { "y", "b", "B" },
                    new[] { "x", "c", "C" },
                    new[] { "y", "c", "C" },
                },
            };

            TreeNode root = Tree.Learn(data, "");

            Assert.AreEqual("Signal", root.Name);
            Assert.IsTrue(root.ChildNodes.All(child => child.IsLeaf));
            CollectionAssert.AreEqual(new[] { "A", "B", "C" }, root.ChildNodes.Select(child => child.Name).ToArray());
        }

        /// <summary>
        ///     The best attribute is the SECOND column, and a subtree still needs the FIRST. Bigtree 0.4.0 built the
        ///     subtable with <c>Skip(TableIndex + 1)</c>, which dropped the first column too and could not separate the
        ///     rows any more.
        /// </summary>
        [TestMethod]
        public void SubtablesKeepTheColumnsLeftOfTheSplit()
        {
            var data = new TrainingData
            {
                Columns = new[] { "A", "B", "Label" },
                Rows = new[]
                {
                    // B = q is pure (No); under B = p the label follows A.
                    new[] { "0", "p", "Yes" },
                    new[] { "1", "p", "No" },
                    new[] { "0", "p", "Yes" },
                    new[] { "0", "q", "No" },
                    new[] { "1", "q", "No" },
                    new[] { "0", "q", "No" },
                    new[] { "1", "q", "No" },
                },
            };

            TreeNode root = Tree.Learn(data, "");
            Assert.AreEqual("B", root.Name);

            var query = new Dictionary<string, string> { ["B"] = "p", ["A"] = "1" };
            Assert.AreEqual("B -- p --> A -- 1 --> NO", Tree.CalculateResult(root, query, ""));
            query = new Dictionary<string, string> { ["B"] = "p", ["A"] = "0" };
            Assert.AreEqual("B -- p --> A -- 0 --> YES", Tree.CalculateResult(root, query, ""));
        }

        /// <summary>Rows that agree on every attribute but not on the label end in a majority leaf instead of an exception.</summary>
        [TestMethod]
        public void ContradictoryRowsEndInAMajorityLeaf()
        {
            var data = new TrainingData
            {
                Columns = new[] { "A", "Label" },
                Rows = new[]
                {
                    new[] { "x", "Yes" },
                    new[] { "x", "No" },
                    new[] { "x", "Yes" },
                },
            };

            TreeNode root = Tree.Learn(data, "");

            Assert.AreEqual("A", root.Name);
            TreeNode leaf = root.ChildNodes.Single();
            Assert.IsTrue(leaf.IsLeaf);
            Assert.AreEqual("Yes", leaf.Name);
        }
    }
}
