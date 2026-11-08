#:project ../Bigtree.Algorithm/Bigtree.Algorithm.csproj
#:property PublishAot=false
#:property NoWarn=NDW012;NDW016;NDW017
// ID3 on noisy random tables: 0.5.0's first port (commit 3226aad: every node re-encoded each of its columns with
// np.unique and counted them with np.bincount; copied below verbatim but for the Strings helper, which is internal to
// the library) against 0.5.0's Tree.Learn (each column encoded once, the tree grown over integer codes). 0.4.0's ID3
// cannot be the baseline: it drops every column left of the split and throws once rows contradict each other or the
// columns run out, which noisy tables always reach. Checks both grow the same tree, gains' bits included, before timing.
//   dotnet run -c Release benchmarks/DecisionTree.cs
using System.Diagnostics;
using System.Text;
using Bigtree.Algorithm.DecisionTree;
using NumSharp;

foreach (var (rows, attributes) in new[] { (1000, 6), (5000, 8), (50000, 8) })
{
    // Attribute c takes c % 3 + 2 values; the label is an XOR of two attributes with 5% of the rows flipped to "yes".
    var random = new Random(3);
    string[] columns = Enumerable.Range(0, attributes).Select(c => "c" + c).Append("label").ToArray();
    string[][] table = Enumerable.Range(0, rows).Select(_ =>
    {
        var row = new string[attributes + 1];
        for (int c = 0; c < attributes; c++)
            row[c] = "v" + random.Next(c % 3 + 2);
        row[attributes] = (row[0] == "v0" ^ row[3] == "v1" || random.NextDouble() < 0.05) ? "yes" : "no";
        return row;
    }).ToArray();
    var data = new TrainingData { Columns = columns, Rows = table };
    bool sameTree = Dump(FirstPort.Learn(data, "")) == Dump(Tree.Learn(data, ""));

    // Warm both, pause so the tier-1 JIT catches up (one warm-up only queues it), warm again, then time interleaved
    // rounds and keep each one's best: a busy machine slows single rounds, never speeds them up.
    double Time(Action run) { var sw = Stopwatch.StartNew(); run(); return sw.Elapsed.TotalMilliseconds; }
    Action firstRun = () => FirstPort.Learn(data, "");
    Action nowRun = () => Tree.Learn(data, "");
    for (int w = 0; w < 2; w++) { firstRun(); nowRun(); }
    Thread.Sleep(300);
    for (int w = 0; w < 2; w++) { firstRun(); nowRun(); }
    double first = double.MaxValue, now = double.MaxValue;
    for (int round = 0; round < 7; round++) { first = Math.Min(first, Time(firstRun)); now = Math.Min(now, Time(nowRun)); }
    Console.WriteLine($"{rows,6} rows x {attributes} attributes: first port {first,8:F1} ms | 0.5.0 {now,7:F1} ms | x{first / now,5:F1} | same tree: {sameTree}");
}

// The tree as text: names, table indices, edges, leaves, every value list and the gains' raw bits.
static string Dump(TreeNode node)
{
    var text = new StringBuilder();
    void Write(TreeNode n, int depth)
    {
        text.Append(' ', depth).Append(n.IsLeaf ? "L " : "N ").Append(n.Name).Append(" @").Append(n.TableIndex).Append(" <").Append(n.Edge).Append('>');
        if (n.NodeAttribute != null)
            text.Append(" [").Append(string.Join(",", n.NodeAttribute.DifferentAttributeNames)).Append("] ")
                .Append(BitConverter.DoubleToInt64Bits(n.NodeAttribute.InformationGain).ToString("X16"));
        text.Append('\n');
        foreach (var child in n.ChildNodes ?? new List<TreeNode>())
            Write(child, depth + 1);
    }
    Write(node, 0);
    return text.ToString();
}

/// <summary>0.5.0's first ID3 port (commit 3226aad), verbatim but for <see cref="Strings"/>.</summary>
static class FirstPort
{
    /// <summary>
    ///     Grows the subtree of a table: the best attribute becomes the node, and each of its values ends in a leaf or
    ///     recurses on a copied subtable without that column (string rows copied at every level, which is part of what
    ///     0.5.0 removed).
    /// </summary>
    /// <param name="data">The (sub)table: attribute columns, then the label column.</param>
    /// <param name="edgeName">The value that leads to this subtree.</param>
    /// <returns>The subtree's root.</returns>
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

    /// <summary>The table's most frequent label, the first in row order on a tie (the leaf of contradictory rows).</summary>
    /// <param name="data">The table, at least one row.</param>
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

    /// <summary>
    ///     Whether the rows with one value of the node's attribute share a label; adds the leaf to the node when they do
    ///     (a scan of every row per value).
    /// </summary>
    /// <param name="root">The node whose attribute is checked; receives the leaf.</param>
    /// <param name="data">The node's table.</param>
    /// <param name="attributeToCheck">The attribute value.</param>
    /// <returns>True when a leaf was added.</returns>
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
    ///     Picks the table's best attribute by information gain, re-encoding every column with NumSharp at this node: a
    ///     dozen NumSharp calls per attribute, whatever the node's size (the cost 0.5.0 removed).
    /// </summary>
    /// <param name="data">The node's table.</param>
    /// <param name="edge">The value that leads to the node.</param>
    /// <returns>The node, its attribute's values in appearance order.</returns>
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
    ///     Numbers a column's values by first appearance: np.unique over a text array built with per-value dtype
    ///     discovery, then two argsorts to turn sorted codes into appearance ranks.
    /// </summary>
    /// <param name="data">The table.</param>
    /// <param name="column">The column index.</param>
    /// <returns>The rows' codes (in the caller's scope), the number of values and their texts in appearance order.</returns>
    [NDScopedCovered]
    private static (NDArray codes, int count, string[] values) Encode(TrainingData data, int column)
    {
        var unique = np.unique(np.array(data.Rows.Select(row => row[column]).ToArray()), return_index: true, return_inverse: true);
        // np.unique numbers the values in sorted order; re-number them by where each first appears.
        NDArray byAppearance = np.argsort(unique.indices);
        NDArray rank = np.argsort(byAppearance);
        string[] values = Strings(unique.values[byAppearance]);
        return (rank[unique.inverse_indices], values.Length, values);
    }

    /// <summary>The Shannon entropy in bits of each row of a count table.</summary>
    /// <param name="counts">A (rows × labels) float64 count table, every row with a positive sum.</param>
    /// <returns>One entropy per row (in the caller's scope).</returns>
    [NDScopedCovered]
    private static NDArray Entropy(NDArray counts)
    {
        NDArray p = counts / np.sum(counts, axis: 1, keepdims: true);
        // log2 as log(p) / log(2), like Math.Log(p, 2); an absent label (p = 0) contributes 0, not 0 * -inf.
        NDArray terms = np.where(p > 0, p * (np.log(p) / Math.Log(2)), 0.0);
        return -np.sum(terms, axis: 1);
    }

    /// <summary>The library's internal NDHelpers.Strings: a 1-D text array's values.</summary>
    /// <param name="text">A 1-D text array.</param>
    /// <returns>Its values, in order.</returns>
    private static string[] Strings(NDArray text)
    {
        var strings = new string[text.size];
        for (long i = 0; i < strings.LongLength; i++)
            strings[i] = (string)text.GetValue(i);
        return strings;
    }
}
