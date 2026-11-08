# Benchmarks

Each file is a .NET 10 file-based app that references the library project; run it from the repository root:

```bash
dotnet run -c Release benchmarks/MultinomialNaiveBayes.cs
dotnet run -c Release benchmarks/BaumWelch.cs
dotnet run -c Release benchmarks/SvmBatchPrediction.cs
```

The baselines are Bigtree 0.4.0's own code, copied verbatim into the scripts, so every comparison runs both versions
in one process on the same data. Ratios are 0.4.0 time / 0.5.0 time (higher is better). Timings are the best of
several rounds after a warm-up that lets the tier-1 JIT compile; the Baum-Welch and SVM scripts interleave their two
contenders round by round, so background load slows both alike. Release builds only: a Debug build distorts both
sides.

Results on 2026-11-08 (Windows 11, .NET 10, NumSharp 0.80.0, managed kernels; the machine was ~25% busy with other
work, which the best-of-N method filters out):

## Multinomial Naive Bayes training (CherubNLP's pass: every label × feature × value through `CalCondProb`)

| sentences × features × labels | 0.4.0 | 0.5.0 | ratio |
|---|---|---|---|
| 300 × 100 × 4 | 64.0 ms | 1.73 ms | ×37 |
| 1,000 × 300 × 5 | 1,997 ms | 9.23 ms | ×216 |
| 2,000 × 500 × 6 | 4,591 ms | 23.6 ms | ×194 |

0.4.0 rebuilt the label's sample matrix on every call (O(labels · rows · features²) per pass); 0.5.0 counts each feature
value once with one matrix product. The log-probabilities are bit-identical.

## Baum-Welch (10 iterations)

| sequences × length, states, symbols | rows × states² | 0.4.0 | 0.5.0 | ratio |
|---|---|---|---|---|
| 10 × 50, 2, 3 | 2,000 | 16.4 ms | 0.28 ms | ×58 |
| 20 × 50, 3, 4 | 9,000 | 15.4 ms | 0.76 ms | ×20 |
| 20 × 100, 3, 4 | 18,000 | 30.9 ms | 1.46 ms | ×21 |
| 40 × 100, 3, 4 | 36,000 | 62.8 ms | 2.57 ms | ×24 |
| 20 × 100, 5, 6 | 50,000 | 92.1 ms | 2.35 ms | ×39 |
| 40 × 100, 4, 6 | 64,000 | 166.4 ms | 4.50 ms | ×37 |
| 80 × 100, 3, 4 | 72,000 | 125.5 ms | 5.10 ms | ×25 |
| 20 × 200, 6, 8 | 144,000 | 280.5 ms | 6.18 ms | ×45 |
| 100 × 200, 8, 16 | 1,280,000 | 2,726 ms | 51.9 ms | ×53 |

0.5.0 runs Rabiner's scaled forward-backward: each time step is normalized by its probability, so the recursions are
multiply-adds where 0.4.0 took an `exp` and a `log1p` per summed term in log space, and the transition statistics
accumulate as outer products instead of a stored (N, N) posterior per time step. Memory follows: 400 sequences × 400
observations × 16 states (3 iterations) peaked at 4.3 GB in 0.5.0's first, NumSharp-tensor E/M step and runs in 43 MB
now (0.42 s instead of 15.2 s). The learned parameters agree with 0.4.0's to rounding (7e-14 here).

## SVM prediction of a whole test set (RBF kernel)

| support vectors, rows × features, classes | per-row (`Predict(Model, Node[])`) | batch (`PredictLabels(Model, NDArray)`) | ratio |
|---|---|---|---|
| 281, 1,000 × 8, 3 | 5.9 ms | 3.9 ms | ×1.5 |
| 988, 5,000 × 20, 4 | 194.9 ms | 66.1 ms | ×2.9 |
| 1,999, 10,000 × 50, 5 | 1,888 ms | 399.7 ms | ×4.7 |

The batch path computes the kernel matrix with one matrix product plus one fused `np.evaluate` pass, every class
pair's decision values with one more product, and the votes with a third; the labels equal the per-row ones.
