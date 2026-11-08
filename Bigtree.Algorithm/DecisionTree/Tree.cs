using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using NumSharp;

namespace Bigtree.Algorithm.DecisionTree
{
    public class Tree
    {
        public TreeNode Root { get; set; }

        public static void Print(TreeNode node, string result)
        {
            if (node?.ChildNodes == null || node.ChildNodes.Count == 0)
            {
                var seperatedResult = result.Split(' ');

                foreach (var item in seperatedResult)
                {
                    if (item.Equals(seperatedResult[0]))
                    {
                        Console.ForegroundColor = ConsoleColor.Magenta;
                    }
                    else if (item.Equals("--") || item.Equals("-->"))
                    {
                        // empty if but better than checking at .ToUpper() and .ToLower() if
                    }
                    else if (item.Equals("YES") || item.Equals("NO"))
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                    }
                    else if (item.ToUpper().Equals(item))
                    {
                        Console.ForegroundColor = ConsoleColor.Cyan;
                    }
                    else
                    {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                    }

                    Console.Write($"{item} ");
                    Console.ResetColor();
                }

                Console.WriteLine();

                return;
            }

            foreach (var child in node.ChildNodes)
            {
                Print(child, result + " -- " + child.Edge.ToLower() + " --> " + child.Name.ToUpper());
            }
        }

        public static void PrintLegend(string headline)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"\n{headline}");
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("Magenta color indicates the root node");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Yellow color indicates an edge");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("Cyan color indicates a node");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Green color indicates a decision");
            Console.ResetColor();
        }

        public static string CalculateResult(TreeNode root, IDictionary<string, string> valuesForQuery, string result)
        {
            var valueFound = false;

            result += root.Name.ToUpper() + " -- ";

            if (root.IsLeaf)
            {
                result = root.Edge.ToLower() + " --> " + root.Name.ToUpper();
                valueFound = true;
            }
            else
            {
                foreach (var childNode in root.ChildNodes)
                {
                    foreach (var entry in valuesForQuery)
                    {
                        if (childNode.Edge.ToUpper().Equals(entry.Value.ToUpper()) && root.Name.ToUpper().Equals(entry.Key.ToUpper()))
                        {
                            valuesForQuery.Remove(entry.Key);

                            return result + CalculateResult(childNode, valuesForQuery, $"{childNode.Edge.ToLower()} --> ");
                        }
                    }
                }
            }

            // if the user entered an invalid attribute
            if (!valueFound)
            {
                result = "Attribute not found";
            }

            return result;
        }

        /// <summary>
        ///     Grows an ID3 decision tree: the attribute with the highest information gain becomes the node, and every
        ///     value of it either ends in a leaf (all its rows share one label) or grows a subtree from those rows
        ///     without that attribute.
        /// </summary>
        /// <param name="data">The training table: attribute columns, then the label column last.</param>
        /// <param name="edgeName">The attribute value that leads to this subtree ("" for the root).</param>
        /// <returns>The subtree's root node.</returns>
        /// <remarks>
        ///     <para><b>How it computes.</b> Every column is encoded ONCE: a NumSharp text array (sized up front, see
        ///     <see cref="CodedTable.Create"/>) that <c>np.unique</c> turns into integer codes. The tree then grows over
        ///     row-index subsets of those codes, counting each attribute's (value × label) table in one pass over the
        ///     node's rows. Re-encoding the text at every node made a node of a few rows pay a dozen NumSharp calls per
        ///     attribute (5,000 rows × 8 attributes took 0.8 s, now 61 ms in a fresh process).</para>
        ///     <para>A subtable keeps every attribute except the one just used. Bigtree 0.4.0 also dropped every column
        ///     to its LEFT (<c>Skip(TableIndex + 1)</c>), which lost attributes whenever the best one was not the
        ///     first; the trees of tables where that never happened are unchanged.</para>
        ///     <para>The entropy is the multi-class Shannon entropy, with <c>log(p) / log(2)</c> as 0.4.0's
        ///     <c>Math.Log(p, 2)</c> computes it and the terms summed in first-appearance order: 0.4.0 split the labels
        ///     into "the first row's label" and "anything else" for the attributes but not for the table, which is the
        ///     same for two labels and inconsistent beyond. Edges keep the order in which the values first appear.</para>
        ///     <para>When the attributes run out before the rows agree on a label (contradictory rows), the subtree is a
        ///     leaf with the most frequent label (the first one in row order on a tie) instead of an exception.</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="data"/> has no rows.</exception>
        public static TreeNode Learn(TrainingData data, string edgeName)
        {
            if (data.Rows.Length == 0)
                throw new ArgumentException("The training table has no rows.", nameof(data));

            var table = CodedTable.Create(data);
            var rows = new int[data.Rows.Length];
            for (int i = 0; i < rows.Length; i++)
                rows[i] = i;
            var attributes = Enumerable.Range(0, data.Columns.Length - 1).ToArray();
            return Grow(table, rows, attributes, edgeName);
        }

        /// <summary>
        ///     The training table with every column encoded once: integer codes per row, and each code's text.
        /// </summary>
        private sealed class CodedTable
        {
            /// <summary>The column names; the last one is the label.</summary>
            public string[] Names;

            /// <summary><c>Codes[column][row]</c>: the row's value in that column, as an index into <see cref="Values"/>.</summary>
            public int[][] Codes;

            /// <summary><c>Values[column][code]</c>: the text of a code.</summary>
            public string[][] Values;

            /// <summary>The label column's index.</summary>
            public int Label => Names.Length - 1;

            /// <summary>Encodes every column of a table with <c>np.unique</c> over a NumSharp text array.</summary>
            /// <param name="data">The table.</param>
            /// <returns>The coded table.</returns>
            /// <remarks>
            ///     Each column becomes a text array sized up front (<see cref="Internal.NDHelpers.Text"/>):
            ///     <c>np.array(string[])</c> alone runs NumPy's dtype discovery on every value, 10 times the cost of the
            ///     copy for the same array.
            /// </remarks>
            /// <exception cref="ArgumentException">A cell is null: NumPy would need an object array for it (NumSharp's
            ///     discovery refuses it), and a sized request would store the text "None" in its place.</exception>
            public static CodedTable Create(TrainingData data)
            {
                using var scope = NDScope.Open();
                int columns = data.Columns.Length;
                var table = new CodedTable { Names = data.Columns, Codes = new int[columns][], Values = new string[columns][] };
                var column = new string[data.Rows.Length];
                for (int c = 0; c < columns; c++)
                {
                    // Checked here rather than in NDHelpers.Text so the message names the row and the column.
                    for (int i = 0; i < column.Length; i++)
                        column[i] = data.Rows[i][c] ?? throw new ArgumentException($"Row {i} has no value in column '{data.Columns[c]}'.", nameof(data));

                    var unique = np.unique(Internal.NDHelpers.Text(column), return_inverse: true);
                    table.Values[c] = Internal.NDHelpers.Strings(unique.values);
                    table.Codes[c] = unique.inverse_indices.astype(np.int32).ToArray<int>();
                }
                return table;
            }
        }

        /// <summary>
        ///     Grows the subtree of a row subset: picks its best attribute and adds one child per value of it (in order of
        ///     first appearance among the rows), a leaf where the value's rows agree on the label.
        /// </summary>
        /// <param name="table">The coded table.</param>
        /// <param name="rows">The subset's row indices, in table order.</param>
        /// <param name="attributes">The attribute columns still available, in table order.</param>
        /// <param name="edge">The attribute value that leads to this subtree.</param>
        /// <returns>The subtree's root.</returns>
        /// <remarks>
        ///     A node reads its rows once for the labels, once per attribute (ranking the values and counting the
        ///     (value × label) table together), and once more to hand each impure value its rows; the chosen attribute's
        ///     table already tells which values are leaves (a single label counted) and how many rows each child gets.
        /// </remarks>
        private static TreeNode Grow(CodedTable table, int[] rows, int[] attributes, string edge)
        {
            if (attributes.Length == 0)
                return new TreeNode(true, MajorityLabel(table, rows), edge);

            int[] labels = table.Codes[table.Label];
            var (labelOrder, labelTotals) = CountByAppearance(labels, rows, table.Values[table.Label].Length, null, null, 1);
            int labelCount = labelTotals.Length;
            double tableEntropy = Entropy(labelTotals, rows.Length);

            int best = -1;
            double bestGain = double.MinValue;
            NodeAttribute bestAttribute = null;
            int[] bestOrder = null;
            double[] bestJoint = null;
            var counts = new double[labelCount];
            for (int a = 0; a < attributes.Length; a++)
            {
                int[] codes = table.Codes[attributes[a]];
                // joint[v, l] = rows whose attribute value is the v-th to appear and whose label is the l-th to appear.
                var (valueOrder, joint) = CountByAppearance(codes, rows, table.Values[attributes[a]].Length, labels, labelOrder, labelCount);
                int valueCount = joint.Length / labelCount;

                // Gain = H(S) - sum over values of |S_v| / |S| * H(S_v), summed in first-appearance order.
                double remainder = 0;
                for (int v = 0; v < valueCount; v++)
                {
                    Array.Copy(joint, v * labelCount, counts, 0, labelCount);
                    double size = 0;
                    foreach (double count in counts)
                        size += count;
                    remainder += size / rows.Length * Entropy(counts, size);
                }

                double gain = tableEntropy - remainder;
                // Strictly greater: the first attribute (in table order) wins a tie, as in 0.4.0.
                if (gain > bestGain)
                {
                    bestGain = gain;
                    best = a;
                    bestOrder = valueOrder;
                    bestJoint = joint;
                    bestAttribute = new NodeAttribute(table.Names[attributes[a]], InAppearanceOrder(table.Values[attributes[a]], valueOrder, valueCount).ToList())
                    {
                        InformationGain = gain,
                    };
                }
            }

            // TableIndex is the attribute's position in this subtree's table, which keeps the remaining columns in order.
            var node = new TreeNode(bestAttribute.Name, best, bestAttribute, edge);
            int[] remaining = attributes.Where((attribute, index) => index != best).ToArray();
            int[] valueCodes = table.Codes[attributes[best]];
            int children = bestAttribute.DifferentAttributeNames.Count;

            // A value whose rows all share one label (one nonzero count in its row of the table) ends in a leaf with that
            // label; any other value gets an exactly sized array for its rows.
            var leafLabel = new int[children];
            var subsets = new int[children][];
            for (int v = 0; v < children; v++)
            {
                int present = 0, only = -1;
                double size = 0;
                for (int l = 0; l < labelCount; l++)
                {
                    double count = bestJoint[v * labelCount + l];
                    if (count > 0)
                    {
                        present++;
                        only = l;
                        size += count;
                    }
                }
                leafLabel[v] = present == 1 ? only : -1;
                if (present > 1)
                    subsets[v] = new int[(int)size];
            }

            // One pass hands every impure value its rows, which stay in table order (the order every count, edge and
            // first-row rule below them reads).
            var filled = new int[children];
            foreach (int row in rows)
            {
                int v = bestOrder[valueCodes[row]];
                if (subsets[v] is int[] subset)
                    subset[filled[v]++] = row;
            }

            var labelByRank = new int[labelCount];
            for (int code = 0; code < labelOrder.Length; code++)
                if (labelOrder[code] >= 0)
                    labelByRank[labelOrder[code]] = code;
            for (int v = 0; v < children; v++)
            {
                string value = bestAttribute.DifferentAttributeNames[v];
                node.ChildNodes.Add(leafLabel[v] >= 0
                    ? new TreeNode(true, table.Values[table.Label][labelByRank[leafLabel[v]]], value)
                    : Grow(table, subsets[v], remaining, value));
            }

            return node;
        }

        /// <summary>
        ///     Ranks the codes present among a row subset in order of first appearance (the order 0.4.0's
        ///     <c>Distinct()</c> gave the edges and the entropy terms) and counts the rows per (rank, label rank) in the
        ///     same pass.
        /// </summary>
        /// <param name="codes">A column's codes.</param>
        /// <param name="rows">The subset's row indices, in table order.</param>
        /// <param name="vocabulary">The column's number of distinct values in the whole table.</param>
        /// <param name="labels">The label codes, or null to count the column's own values (one count per rank).</param>
        /// <param name="labelOrder">The label codes' appearance ranks among the same rows (null with <paramref name="labels"/>).</param>
        /// <param name="labelCount">The number of label ranks (1 when <paramref name="labels"/> is null).</param>
        /// <returns>
        ///     <c>order[code]</c> = the code's appearance rank (-1 when absent from the subset), and the counts as a
        ///     (present ranks × <paramref name="labelCount"/>) row-major table.
        /// </returns>
        private static (int[] order, double[] counts) CountByAppearance(int[] codes, int[] rows, int vocabulary, int[] labels, int[] labelOrder, int labelCount)
        {
            var order = new int[vocabulary];
            Array.Fill(order, -1);
            // Sized for every code of the table, since the ranks are only known as the rows come; trimmed to the present ones.
            var counts = new double[vocabulary * labelCount];
            int present = 0;
            foreach (int row in rows)
            {
                int code = codes[row];
                int rank = order[code];
                if (rank < 0)
                    rank = order[code] = present++;
                counts[rank * labelCount + (labels is null ? 0 : labelOrder[labels[row]])]++;
            }

            if (present < vocabulary)
                Array.Resize(ref counts, present * labelCount);
            return (order, counts);
        }

        /// <summary>The texts of the codes present in a subset, in appearance order.</summary>
        /// <param name="values">The column's texts by code.</param>
        /// <param name="order">The codes' appearance ranks (see <see cref="CountByAppearance"/>).</param>
        /// <param name="count">The number of codes present.</param>
        /// <returns>One text per present value.</returns>
        private static string[] InAppearanceOrder(string[] values, int[] order, int count)
        {
            var result = new string[count];
            for (int code = 0; code < order.Length; code++)
                if (order[code] >= 0)
                    result[order[code]] = values[code];
            return result;
        }

        /// <summary>The Shannon entropy in bits of one row of counts.</summary>
        /// <param name="counts">The counts, in the order their terms are summed.</param>
        /// <param name="total">Their sum (positive).</param>
        /// <returns><c>-sum(p * log2(p))</c> with <c>p = count / total</c> and 0 log 0 = 0.</returns>
        private static double Entropy(double[] counts, double total)
        {
            double sum = 0;
            foreach (double count in counts)
            {
                double p = count / total;
                // log2 as log(p) / log(2), like 0.4.0's Math.Log(p, 2); an absent label contributes 0, not 0 * -inf.
                if (p > 0)
                    sum += p * (Math.Log(p) / Math.Log(2));
            }
            return -sum;
        }

        /// <summary>The most frequent label of a row subset, the first in row order on a tie.</summary>
        /// <param name="table">The coded table.</param>
        /// <param name="rows">The subset's row indices, at least one.</param>
        /// <returns>The label's text.</returns>
        private static string MajorityLabel(CodedTable table, int[] rows)
        {
            int[] labels = table.Codes[table.Label];
            int winner = rows
                .Select((row, index) => (label: labels[row], index))
                .GroupBy(entry => entry.label)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.First().index)
                .First().Key;
            return table.Values[table.Label][winner];
        }
    }
}