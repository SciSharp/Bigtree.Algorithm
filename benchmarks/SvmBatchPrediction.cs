#:project ../Bigtree.Algorithm/Bigtree.Algorithm.csproj
#:package NumSharp.Build@0.80.0
#:property PublishAot=false
// SVM prediction of a whole test set: the per-row libsvm path (Prediction.Predict(Model, Node[]) for every row)
// against 0.5.0's batch path (Prediction.PredictLabels(Model, NDArray): one NumSharp kernel matrix and one
// matrix-vector product per class pair). Checks both give the same labels before timing. NumSharp.Build weaves the
// [NDScoped] methods below, so each configuration's and each batch run's arrays return to NumSharp's pool.
//   dotnet run -c Release benchmarks/SvmBatchPrediction.cs
using System.Diagnostics;
using Bigtree.Algorithm.SVM;
using NumSharp;

foreach (var (train, test, features, classes) in new[] { (300, 1000, 8, 3), (1000, 5000, 20, 4), (2000, 10000, 50, 5) })
    Benchmark.Run(train, test, features, classes);

/// <summary>The benchmark's two contenders, timed for one problem size.</summary>
static class Benchmark
{
    /// <summary>Trains on a random problem, checks both paths agree, then times them interleaved.</summary>
    /// <param name="train">The training samples.</param>
    /// <param name="test">The rows to predict.</param>
    /// <param name="features">The features per sample.</param>
    /// <param name="classes">The class count.</param>
    [NDScoped]
    public static void Run(int train, int test, int features, int classes)
    {
        var rng = np.random.default_rng(1);
        NDArray Xtrain = rng.normal(size: new Shape(train, features));
        NDArray ytrain = np.floor(np.abs(Xtrain[":, 0"] * 2 + Xtrain[":, 1"])).astype(np.int64) % classes;
        NDArray Xtest = rng.normal(size: new Shape(test, features));
        Model model = Training.Train(Problem.FromNDArray(Xtrain, ytrain), new Parameter { Gamma = 1.0 / features, C = 4 });
        Node[][] rows = Problem.FromNDArray(Xtest, np.zeros(new Shape(test))).X;

        double[] perRowLabels = rows.Select(model.Predict).ToArray();
        double[] batchLabels = model.PredictLabels(Xtest).ToArray<double>();
        int disagree = perRowLabels.Where((label, i) => label != batchLabels[i]).Count();

        // Warm both paths, pause so the tier-1 JIT catches up (one warm-up only queues it), warm again, then time
        // interleaved rounds and keep each path's best: a busy machine slows single rounds, never speeds them up.
        double Time(Action run) { var sw = Stopwatch.StartNew(); run(); return sw.Elapsed.TotalMilliseconds; }
        Action perRowRun = () => { foreach (Node[] row in rows) model.Predict(row); };
        Action batchRun = () => PredictBatch(model, Xtest);
        for (int w = 0; w < 3; w++) { perRowRun(); batchRun(); }
        Thread.Sleep(300);
        for (int w = 0; w < 3; w++) { perRowRun(); batchRun(); }
        double perRow = double.MaxValue, batch = double.MaxValue;
        for (int round = 0; round < 10; round++) { perRow = Math.Min(perRow, Time(perRowRun)); batch = Math.Min(batch, Time(batchRun)); }
        Console.WriteLine($"{model.SupportVectorCount,5} SVs, {test,6} rows x {features,3} features, {classes} classes: per-row {perRow,9:F1} ms | batch {batch,8:F1} ms | x{perRow / batch,5:F1} | labels differing: {disagree}");
    }

    /// <summary>One timed batch prediction; its result and temporaries return to the pool when it returns.</summary>
    /// <param name="model">The trained model.</param>
    /// <param name="Xtest">The rows to predict.</param>
    [NDScoped]
    private static void PredictBatch(Model model, NDArray Xtest) => model.PredictLabels(Xtest);
}
