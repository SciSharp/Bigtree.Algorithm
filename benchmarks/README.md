# Benchmarks

Each file is a .NET 10 file-based app that references the library project; run it from the repository root:

```bash
dotnet run -c Release benchmarks/MultinomialNaiveBayes.cs
dotnet run -c Release benchmarks/BaumWelch.cs
dotnet run -c Release benchmarks/SvmBatchPrediction.cs
dotnet run -c Release benchmarks/DecisionTree.cs
```

The baselines are Bigtree 0.4.0's own code, copied verbatim into the scripts, so every comparison runs both versions
in one process on the same data; the decision tree's is 0.5.0's first port, since 0.4.0's ID3 cannot grow those
trees (see its section). Ratios are baseline time / 0.5.0 time (higher is better). Timings are the best of several
rounds after a warm-up that lets the tier-1 JIT compile; the Baum-Welch, SVM and decision-tree scripts interleave their
two contenders round by round, so background load slows both alike. Release builds only: a Debug build distorts both
sides.

Results on 2026-11-08, the Naive Bayes, SVM and decision-tree tables on 2026-11-09 (Windows 11, .NET 10, NumSharp
0.80.0, managed kernels; the machine was ~25% busy with other work, which the best-of-N method filters out):

## Multinomial Naive Bayes training (CherubNLP's pass: every label × feature × value through `CalCondProb`)

| sentences × features × labels | 0.4.0 | 0.5.0 | ratio |
|---|---|---|---|
| 300 × 100 × 4 | 33.1 ms | 0.47 ms | ×71 |
| 1,000 × 300 × 5 | 1,415 ms | 2.37 ms | ×598 |
| 2,000 × 500 × 6 | 4,184 ms | 4.79 ms | ×874 |

(2026-11-09; the load on the machine moved 0.4.0's times by up to 1.5× between runs: ratios ×47 to ×71, ×574 to ×598,
×874 to ×1,147 over two runs.) 0.4.0 rebuilt the label's sample matrix on every call (O(labels · rows · features²) per
pass); 0.5.0 counts each requested value once, in one SIMD pass over the samples' own vectors into a (labels × features)
table, and NumSharp turns the table into log-probabilities. The log-probabilities are bit-identical. Memory follows: a
training pass over 20,000 samples × 2,000 features peaked at 1,358 MB with 0.5.0's first, matrix-product count (four
copies of the samples per value) and at 376 MB now, 320 MB of which are the samples' own vectors (0.95-1.95 s →
0.12-0.15 s).

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
| 281, 1,000 × 8, 3 | 6.2 ms | 4.0 ms | ×1.6 |
| 988, 5,000 × 20, 4 | 197.1 ms | 69.6 ms | ×2.8 |
| 1,999, 10,000 × 50, 5 | 2,379 ms | 341.5 ms | ×7.0 |

The batch path computes the kernel matrix with one matrix product plus one fused `np.evaluate` pass, every class
pair's decision values with one more product, and the votes with a third; the labels equal the per-row ones. It runs
in blocks of 2^22 kernel entries (32 MB), each block's temporaries returned before the next: 20,000 rows × 1,960
support vectors peaked at 186 MB, against 696 MB with the whole kernel matrix at once, in the same time.

## ID3 decision tree (noisy XOR tables: attribute c takes c % 3 + 2 values, 5% of the labels flipped)

| rows × attributes | 0.5.0's first port | 0.5.0 | ratio |
|---|---|---|---|
| 1,000 × 6 | 27.7 ms | 1.9 ms | ×15 |
| 5,000 × 8 | 190.9 ms | 2.7 ms | ×70 |
| 50,000 × 8 | 576.4 ms | 26.3 ms | ×22 |

0.4.0's ID3 is no baseline: it drops every column left of the chosen one and throws once rows contradict each other or
the columns run out, which noisy tables always reach. 0.5.0's first port fixed that and re-encoded every column with
NumSharp at every node, a dozen NumSharp calls per attribute however few rows the node had. 0.5.0 encodes each column
once (`np.unique` over a text array sized up front: NumSharp 0.80.0's `np.array(string[])` discovers the dtype value by
value, 10 times the cost of the copy) and grows the tree over integer codes, reading a node's rows once per attribute.
Both grow the same tree, gains bit for bit (also checked on 336 random tables of 1 to 1,500 rows). In a fresh process,
one `Learn` of 5,000 rows takes 48 ms instead of 0.83 s, 50,000 rows 110 ms instead of 1.6 s, peaking at 53 MB and
85 MB instead of 90 MB and 189 MB.
