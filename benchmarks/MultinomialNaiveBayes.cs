#:project ../Bigtree.Algorithm/Bigtree.Algorithm.csproj
#:property PublishAot=false
#:property NoWarn=NDW012;NDW016;NDW017
// CherubNLP's Naive Bayes training pass (every label x feature x value 0/1 through CalCondProb) on Bigtree 0.4.0's
// MultinomiaNaiveBayes (copied verbatim below as Legacy) and on 0.5.0's, which counts each value in one SIMD pass over
// the samples and turns the counts into log-probabilities with NumSharp.
//   dotnet run -c Release benchmarks/MultinomialNaiveBayes.cs
using System.Diagnostics;
using Bigtree.Algorithm.Bayes;
using Bigtree.Algorithm.Statistics;

foreach (var (N, F, L) in new[] { (300, 100, 4), (1000, 300, 5), (2000, 500, 6) })
{
    var random = new Random(7);
    var featureSets = Enumerable.Range(0, N)
        .Select(i => Tuple.Create("label" + random.Next(L), Enumerable.Range(0, F).Select(f => random.NextDouble() < 0.1 ? 1.0 : 0.0).ToArray()))
        .ToList();
    var labelDist = featureSets.GroupBy(x => x.Item1).Select(g => new Probability { Value = g.Key, Freq = g.Count() }).OrderBy(p => p.Value).ToList();

    double Train(Func<string, int, double, double> condProb)
    {
        var sw = Stopwatch.StartNew();
        double checksum = 0;
        foreach (var label in labelDist)
            for (int x = 0; x < F; x++)
                foreach (double v in new[] { 0.0, 1.0 })
                    checksum += condProb(label.Value, x, v);
        sw.Stop();
        GC.KeepAlive(checksum);
        return sw.Elapsed.TotalMilliseconds;
    }

    // Each round trains a fresh classifier, as CherubNLP's Train does; the first two rounds warm the JIT. 0.5.0's
    // rounds take milliseconds, so it gets more of them: a busy machine slows single rounds, never speeds them up.
    double best(Func<Func<string, int, double, double>> factory, int rounds) => Enumerable.Range(0, rounds).Select(_ => Train(factory())).Skip(2).Min();
    double legacy = best(() => { var nb = new Legacy { LabelDist = labelDist, FeatureSet = featureSets }; return (y, x, v) => nb.CalCondProb(x, y, v); }, 5);
    double current = best(() => { var nb = new MultinomiaNaiveBayes { LabelDist = labelDist, FeatureSet = featureSets }; return (y, x, v) => nb.CalCondProb(x, y, v); }, 25);
    Console.WriteLine($"{N,5} sentences x {F,4} features x {L} labels: 0.4.0 {legacy,10:F2} ms | 0.5.0 {current,8:F2} ms | x{legacy / current,6:F0}");
}

/// <summary>Bigtree 0.4.0's MultinomiaNaiveBayes conditional probability, verbatim (commit dfb20dd).</summary>
class Legacy
{
    /// <summary>The labels and their frequencies.</summary>
    public List<Probability> LabelDist { get; set; }

    /// <summary>The training samples.</summary>
    public List<Tuple<string, double[]>> FeatureSet { get; set; }

    /// <summary>The additive smoothing, 0.4.0's default.</summary>
    private double alpha { get; set; } = 0.5;

    /// <summary>0.4.0's conditional log-probability: filter the label's samples, copy them into a matrix, count.</summary>
    /// <param name="x">The feature.</param>
    /// <param name="Y">The label.</param>
    /// <param name="feature">The value.</param>
    /// <returns>log((n_yx + alpha) / (n_y + n_y * alpha)).</returns>
    public double CalCondProb(int x, string Y, double feature)
    {
        var featuresIfY = FeatureSet.Where(fd => fd.Item1 == Y).ToList();
        var matrix = ConstructMatrix(featuresIfY);
        int freq = 0;
        for (int y = 0; y < featuresIfY.Count; y++)
            if (matrix[y, x] == feature)
                freq++;
        int Nyk = featuresIfY.Count;
        int n = featuresIfY.Count;
        int Nykx = freq;
        return Math.Log((Nykx + alpha) / (Nyk + n * alpha));
    }

    /// <summary>0.4.0's per-call copy of a label's samples into a matrix.</summary>
    /// <param name="featuresIfY">The label's samples.</param>
    /// <returns>The samples as a (samples, features) matrix.</returns>
    private double[,] ConstructMatrix(List<Tuple<string, double[]>> featuresIfY)
    {
        var featureCount = featuresIfY[0].Item2.Length;
        double[,] matrix = new double[featuresIfY.Count, featureCount];
        for (int y = 0; y < featuresIfY.Count; y++)
            for (int x = 0; x < featureCount; x++)
                matrix[y, x] = featuresIfY[y].Item2[x];
        return matrix;
    }
}
