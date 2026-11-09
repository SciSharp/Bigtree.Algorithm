# Bigtree.Algorithm

Machine Learning library in .NET, computed on [NumSharp](https://github.com/SciSharp/NumSharp) 0.80.0 (net8.0 and
net10.0).

### Implemented Models

| Model | Classes | How it computes |
|---|---|---|
| SVM (Support Vector Machines) | `SVM.Training`, `Prediction`, `Model`, `Problem` | libsvm 3.x's SMO solver; batch prediction of an `NDArray` as a NumSharp kernel matrix |
| CRF (Conditional Random Field) | `CRFLite.CRFEncoder`, `CRFDecoder` | CRFSharp (L-BFGS training, Viterbi decoding) |
| KNN | `KNN.KnnModel` | distance matrix by one matrix product, `np.argpartition`, one-hot votes |
| Decision Tree | `DecisionTree.Tree` | ID3; each column encoded once by `np.unique` over a NumSharp text array, the tree grown over the integer codes |
| HMM (Hidden Markov Model) | `HiddenMarkovModel.*` | forward/backward/Viterbi; Baum-Welch on Rabiner's scaled forward-backward |
| Bayes | `Bayes.GaussianNaiveBayes`, `BernoulliNaiveBayes`, `MultinomialNaiveBayes` | per-class reductions and matrix products; the multinomial one counts each feature value in one SIMD pass over its samples' vectors |
| Linear Regression | `LinearRegression.LinearRegressionModel` | centered normal equations solved by `np.linalg.solve` (managed) |
| Artificial Neural Network | `NeuralNetwork.NetworkModel` | one weight matrix per layer, mini-batch backpropagation as matrix products and compiled `NDExpr` programs |

The NumSharp estimators follow scikit-learn's semantics with .NET names: `Fit(X, y)`, `Predict(X)`,
`PredictProba(X)`, `Score(X, y)`, fitted attributes as properties (`Theta`, `Var`, `Coef`, ...). Labels can be any
NumSharp dtype, text included.

```csharp
using Bigtree.Algorithm;
using Bigtree.Algorithm.Bayes;
using Bigtree.Algorithm.KNN;
using Bigtree.Algorithm.NeuralNetwork;
using Bigtree.Algorithm.SVM;
using NumSharp;

var (X, y) = Utils.ReadCsv("data/seeds_dataset.csv");       // np.loadtxt + np.unique
X = Utils.Normalize(X);

using var bayes = new GaussianNaiveBayes().Fit(X, y);
NDArray proba = bayes.PredictProba(X);

using var knn = new KnnModel(k: 5).Fit(X, y);
var (distances, indices) = knn.KNeighbors(X["0:3"]);

using var network = new NetworkModel();
network.AddLayer(new NeuralLayer(7, "INPUT"));
network.AddLayer(new NeuralLayer(5, "HIDDEN"));
network.AddLayer(new NeuralLayer(3, "OUTPUT"));
network.Build(seed: 1);
network.Train(X, y, iterations: 1000, learningRate: 1.5, batchSize: 8);
NDArray predicted = network.Predict(X);

Model svm = Training.Train(Problem.FromNDArray(X, y), new Parameter { Gamma = 1.0 / 7 });
NDArray labels = svm.PredictLabels(X);
```

`Bigtree.Algorithm.Console` runs every model on the seeds dataset.

### Same numbers as scikit-learn

`Bigtree.Algorithm.Tests.Oracle` embeds CPython through `NumSharp.Interop.pythonnet` and compares each estimator
with scikit-learn 1.8 (or a NumPy / SciPy reference where scikit-learn has no twin) on the same arrays:

- `GaussianNaiveBayes` is **bit-identical** to `GaussianNB` (`theta_`, `var_`, `class_prior_`, `epsilon_`, every
  prediction output): it repeats scikit-learn's operations, and NumSharp's reductions and ufuncs are NumPy's.
- `BernoulliNaiveBayes`: counts and log-probabilities bit-identical; probabilities to 1e-10 (scikit-learn's matrix
  product runs on BLAS).
- `KnnModel`: the same neighbours, vote fractions and labels as `KNeighborsClassifier`; distances to 1e-12.
- `LinearRegressionModel`: `LinearRegression`'s coefficients to 1e-10 (SVD there, normal equations here).
- The SVM: the same support vectors, dual coefficients, intercepts and one-vs-one decision values as `SVC` (to
  1e-9); on the unit tests' problems every SVM type × kernel scores exactly `SVC` / `NuSVC`'s accuracy.
- `NetworkModel`: NumPy's mini-batch backpropagation from the same weights, to 1e-12; Baum-Welch: a NumPy / SciPy
  log-space implementation, to 1e-9; ID3's gains: `scipy.stats.entropy`; `Utils.CrossValFolds`: NumPy's
  `default_rng(seed).permutation`, element for element.

Run it with an interpreter that has NumPy, SciPy and scikit-learn:

```bash
NUMSHARP_PYTHONNET_PYTHON=/path/to/python dotnet test Bigtree.Algorithm.Tests.Oracle
```

Without the variable the oracle tests report Inconclusive (and fail under `CI=true`). The unit tests need no Python:

```bash
dotnet test Bigtree.Algorithm.UnitTest
```

### Performance

`benchmarks/` compares 0.5.0 with 0.4.0's own code, in one process: Multinomial Naive Bayes training ×47 to ×1,147,
Baum-Welch ×20 to ×58 (and a 400 × 400 × 16-state run in 43 MB instead of 4.3 GB), SVM batch prediction ×1.6 to ×7.0;
and ID3 ×15 to ×70 against 0.5.0's first port (0.4.0's ID3 cannot grow those trees). See
[benchmarks/README.md](benchmarks/README.md).

NumSharp spends a fixed 1.5-3 µs per call, so the estimators restructure their algorithms into whole-array
operations instead of calling NumSharp per element or per sample. Where an algorithm is inherently sequential over
small vectors (HMM recursions, the SMO solver, CRF decoding, growing a tree over coded columns) it stays a scalar loop.
Element-wise formulas run as one fused `np.evaluate` pass (KNN distances, SVM kernels, the input finiteness checks),
and the neural network's (sigmoid, deltas, weight update) and Gaussian Naive Bayes' variance-weighted distance as
`NDExpr` programs compiled once. The bits stay those of the ufunc chains they replace: the network trains to the same
weights with 11 NumSharp calls per step instead of about 24 (1.7 times faster at batch size 32, 3 times at batch size
1), and Gaussian Naive Bayes stays bit-identical to scikit-learn with its distances 3.75 times faster.

### Memory

NumSharp arrays live in unmanaged memory. The estimators implement `IDisposable` and own their fitted arrays: dispose
a model (or let the finalizer do it) to return the buffers to NumSharp's pool, and never dispose an array a model
exposes. Every array a method returns belongs to the caller. Inside the library, `[NDScoped]` methods (woven at build
time by `NumSharp.Build`) return their temporaries to the pool on exit.

A scope holds every temporary until its method returns, so a method that loops over classes or would build a
(queries × training samples) matrix opens one scope per class or per block instead: KNN and SVM batch prediction run
in blocks of 2^22 distances or kernel entries (32 MB) and write each block into the result, linear regression sums its
normal equations over 32 MB blocks of centered rows instead of centering a whole copy of `X`, and Gaussian Naive Bayes
frees each class's temporaries before the next. KNN on 8,000 queries × 8,000 training samples peaks at 170 MB instead
of 3.0 GB (and runs twice as fast), Gaussian Naive Bayes on 100,000 × 50 with 10 classes at 280 MB instead of 1.4 GB,
a linear regression fit of 200,000 × 100 at 277 MB instead of 385 MB (160 MB of it the input).

### Changes from 0.4.0

- net8.0 / net10.0 and NumSharp 0.80.0 (from netstandard2.0 and NumSharp 0.30.0).
- `MultinomiaNaiveBayes` and `MultinomiaNaiveBayesModel` are spelled `MultinomialNaiveBayes` and
  `MultinomialNaiveBayesModel`; their members, the SVM types, `Probability`, `Feature`, `CRFLite` and the `Shuffle` /
  `Split` extensions keep their API. CherubNLP compiles once it targets net8.0 and uses the two new names (three lines);
  its saved JSON models load unchanged (the property names did not change).
- `NetworkModel` holds weight matrices: `Neuron` and `Dendrite` are gone, `NeuralLayer` is `(Count, Name)`,
  `Train` takes a `batchSize`, `Build` a `seed`. Its two backpropagation bugs are fixed.
- `Utils.CrossValFolds(N, n_folds, seed)` returns `NDArray[]` folds (0.4.0 returned none); `ReadCsv` numbers labels in
  ascending order; `Utils.Normalize` replaces NumSharp 0.30's `NDArray.Normalize()`.
- SVM: `RangeTransform` serves 1-based (libsvm) and 0-based features; polynomial-kernel predictions use `Gamma` as
  training does; `Model.Write` keeps full precision; no false "weight not found" warnings; NDArray entry points.
- HMM: the scaled forward/backward pair works (`++N` loop, missing first scale); Baum-Welch's convergence value
  covers every sequence.
- Decision tree: multi-class entropy; subtables keep the attributes left of the split; contradictory rows end in a
  majority leaf instead of an exception; a null cell raises `ArgumentException` naming its row and column.
