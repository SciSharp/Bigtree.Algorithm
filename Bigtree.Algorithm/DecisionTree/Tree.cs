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
        ///     <para>A subtable keeps every attribute except the one just used. Bigtree 0.4.0 also dropped every column
        ///     to its LEFT (<c>Skip(TableIndex + 1)</c>), which lost attributes whenever the best one was not the
        ///     first; the trees of tables where that never happened are unchanged.</para>
        ///     <para>When the attributes run out before the rows agree on a label (contradictory rows), the subtree is a
        ///     leaf with the most frequent label (the first one in row order on a tie) instead of an exception.</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="data"/> has no rows.</exception>
        public static TreeNode Learn(TrainingData data, string edgeName)
        {
            if (data.Rows.Length == 0)
                throw new ArgumentException("The training table has no rows.", nameof(data));
            if (data.Columns.Length < 2)
                return new TreeNode(true, MajorityLabel(data), edgeName);

            var root = GetRootNode(data, edgeName);

            foreach (var item in root.NodeAttribute.DifferentAttributeNames)
            {
                // if a leaf, leaf will be added in this method
                var isLeaf = CheckIfIsLeaf(root, data, item);

                // make a recursive call as long as the node is not a leaf
                if (!isLeaf)
                {
                    var reducedTable = new TrainingData
                    {
                        Columns = data.Columns.Where((column, index) => index != root.TableIndex).ToArray(),
                        Rows = data.Rows.Where(x => x[root.TableIndex].Equals(item))
                                .Select(x => x.Where((value, index) => index != root.TableIndex).ToArray())
                                .ToArray()
                    };

                    root.ChildNodes.Add(Learn(reducedTable, item));
                }
            }

            return root;
        }

        /// <summary>The most frequent label of a table, the first in row order on a tie.</summary>
        /// <param name="data">A table with at least one row; its last column is the label.</param>
        /// <returns>The label.</returns>
        private static string MajorityLabel(TrainingData data)
        {
            int labelColumn = data.Columns.Length - 1;
            return data.Rows
                .Select((row, index) => (label: row[labelColumn], index))
                .GroupBy(entry => entry.label)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.First().index)
                .First().Key;
        }

        private static bool CheckIfIsLeaf(TreeNode root, TrainingData data, string attributeToCheck)
        {
            var isLeaf = true;
            var allEndValues = new List<string>();

            // get all leaf values for the attribute in question
            for (var i = 0; i < data.Rows.Length; i++)
            {
                if (data.Rows[i][root.TableIndex].ToString().Equals(attributeToCheck))
                {
                    allEndValues.Add(data.Rows[i][data.Columns.Length - 1].ToString());
                }
            }

            // check whether all elements of the list have the same value
            if (allEndValues.Count > 0 && allEndValues.Any(x => x != allEndValues[0]))
            {
                isLeaf = false;
            }

            // create leaf with value to display and edge to the leaf
            if (isLeaf)
            {
                root.ChildNodes.Add(new TreeNode(true, allEndValues[0], attributeToCheck));
            }

            return isLeaf;
        }

        /// <summary>
        ///     Picks the attribute with the highest information gain over the table's label column; its values (in
        ///     order of first appearance) become the node's edges.
        /// </summary>
        /// <param name="data">The training table, label column last; at least one attribute column.</param>
        /// <param name="edge">The attribute value that leads to this node.</param>
        /// <returns>The node of the best attribute; the first attribute wins a tie.</returns>
        /// <remarks>
        ///     <para>Each column becomes a NumSharp text array that <c>np.unique</c> encodes into value codes, so one
        ///     <c>np.bincount</c> of <c>valueCode * labels + labelCode</c> counts the whole (value × label) table of an
        ///     attribute in a single pass. Bigtree 0.4.0 compared strings row by row once per distinct value.</para>
        ///     <para>The entropy is the multi-class Shannon entropy. 0.4.0 split the labels into "the first row's label"
        ///     and "anything else" for the attributes but not for the table, which is the same for two labels and
        ///     inconsistent beyond. Logarithms are taken as <c>log(p) / log(2)</c>, the way 0.4.0's
        ///     <c>Math.Log(p, 2)</c> computes them, so two-label tables produce the same gains.</para>
        /// </remarks>
        private static TreeNode GetRootNode(TrainingData data, string edge)
        {
            using var scope = NDScope.Open();
            int labelColumn = data.Columns.Length - 1;
            int rows = data.Rows.Length;
            var (labelCodes, labelCount, _) = Encode(data, labelColumn);
            double tableEntropy = Entropy(np.bincount(labelCodes, minlength: labelCount).astype(np.float64).reshape(1, -1)).GetDouble(0);

            var attributes = new List<NodeAttribute>();
            var highestInformationGainIndex = -1;
            var highestInformationGain = double.MinValue;
            for (var i = 0; i < labelColumn; i++)
            {
                var (codes, valueCount, values) = Encode(data, i);
                // joint[v, l] = the rows whose attribute has value v and whose label is l.
                NDArray joint = np.bincount(codes * labelCount + labelCodes, minlength: valueCount * labelCount)
                    .astype(np.float64).reshape(valueCount, labelCount);
                NDArray sizes = np.sum(joint, axis: 1);
                // Gain = H(S) - sum over values of |S_v| / |S| * H(S_v).
                double remainder = np.sum(sizes / rows * Entropy(joint)).GetDouble();
                var attribute = new NodeAttribute(data.Columns[i], values.ToList()) { InformationGain = tableEntropy - remainder };
                attributes.Add(attribute);

                if (attribute.InformationGain > highestInformationGain)
                {
                    highestInformationGain = attribute.InformationGain;
                    highestInformationGainIndex = i;
                }
            }

            return new TreeNode(attributes[highestInformationGainIndex].Name, highestInformationGainIndex, attributes[highestInformationGainIndex], edge);
        }

        /// <summary>
        ///     Encodes one column of the table as integer codes, numbering its distinct values in order of first
        ///     appearance (the order 0.4.0's <c>Distinct()</c> gave the tree's edges).
        /// </summary>
        /// <param name="data">The table.</param>
        /// <param name="column">The column index.</param>
        /// <returns>
        ///     <c>codes</c>: an int64 code per row; <c>count</c>: the number of distinct values; <c>values</c>: the
        ///     distinct values, code order.
        /// </returns>
        [NDScopedCovered]
        private static (NDArray codes, int count, string[] values) Encode(TrainingData data, int column)
        {
            var unique = np.unique(np.array(data.Rows.Select(row => row[column]).ToArray()), return_index: true, return_inverse: true);
            // np.unique numbers the values in sorted order; re-number them by where each first appears.
            NDArray byAppearance = np.argsort(unique.indices);
            NDArray rank = np.argsort(byAppearance);
            string[] values = Internal.NDHelpers.Strings(unique.values[byAppearance]);
            return (rank[unique.inverse_indices], values.Length, values);
        }

        /// <summary>The Shannon entropy in bits of every row of a count table.</summary>
        /// <param name="counts">A float64 (groups, labels) table of counts; every row has a positive sum.</param>
        /// <returns>A float64 vector with each row's <c>-sum(p * log2(p))</c>, where <c>p</c> is the row normalized and 0 log 0 = 0.</returns>
        [NDScopedCovered]
        private static NDArray Entropy(NDArray counts)
        {
            NDArray p = counts / np.sum(counts, axis: 1, keepdims: true);
            // log2 as log(p) / log(2), like Math.Log(p, 2); an absent label (p = 0) contributes 0, not 0 * -inf.
            NDArray terms = np.where(p > 0, p * (np.log(p) / Math.Log(2)), 0.0);
            return -np.sum(terms, axis: 1);
        }
    }
}