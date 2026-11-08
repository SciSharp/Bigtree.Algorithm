using Bigtree.Algorithm.Internal;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Bigtree.Algorithm.NeuralNetwork
{
    /// <summary>
    ///     A fully connected feed-forward network of sigmoid layers without biases, trained by backpropagation with
    ///     mini-batch gradient descent. Every layer is one NumSharp weight matrix, and a batch moves through the
    ///     network as matrix products.
    /// </summary>
    /// <remarks>
    ///     <para><b>Usage.</b> Add the layers (input first), <see cref="Build"/>, <see cref="Train"/>, then
    ///     <see cref="Predict"/>:</para>
    ///     <code>
    ///     using var model = new NetworkModel();
    ///     model.Layers.Add(new NeuralLayer(7, "INPUT"));
    ///     model.Layers.Add(new NeuralLayer(5, "HIDDEN"));
    ///     model.Layers.Add(new NeuralLayer(3, "OUTPUT"));
    ///     model.Build(seed: 1);
    ///     model.Train(X, y, iterations: 1000, learningRate: 0.6);
    ///     NDArray predicted = model.Predict(X);
    ///     </code>
    ///     <para><b>The algorithm.</b> The classic "network from scratch" (as in ankonzoid/NN-scratch, which the seeds
    ///     test ports): sigmoid activations, the squared-error delta <c>(target - out) * out * (1 - out)</c>, weights
    ///     initialized uniformly in [0, 1), batches taken in row order every epoch. A batch's update is the MEAN of its
    ///     samples' updates, computed at the weights the batch started with, so <c>batchSize: 1</c> is exactly the
    ///     per-sample stochastic gradient descent of Bigtree 0.4.0 with its two bugs fixed: the hidden error now sums
    ///     the next layer's deltas only, and each layer's update now uses its own input (the previous layer's output)
    ///     instead of the network input.</para>
    ///     <para><b>Batch size is a speed choice.</b> NumSharp spends a fixed 1.5-3 µs per call, so per-sample
    ///     training on a small network is dominated by that cost (the 7-5-3 seeds network measured about 80 times
    ///     slower than plain loops at batch size 1); a batch of 32 rows divides it by 32. Larger batches make fewer,
    ///     averaged updates per epoch, so they may need more epochs or a larger learning rate.</para>
    ///     <para><b>Ownership.</b> <see cref="Weights"/> belong to the model: never dispose them. Dispose the model (or
    ///     let the finalizer do it) to return their buffers to NumSharp's pool.</para>
    /// </remarks>
    public class NetworkModel : IDisposable
    {
        /// <summary>The layers, input first and output last; each says how many neurons it has.</summary>
        public List<NeuralLayer> Layers { get; set; }

        /// <summary>
        ///     The weights between consecutive layers after <see cref="Build"/>: <c>Weights[l]</c> has shape
        ///     (<c>Layers[l + 1].Count</c>, <c>Layers[l].Count</c>), row <c>j</c> holding the input weights of neuron
        ///     <c>j</c> of layer <c>l + 1</c>. <c>null</c> before <see cref="Build"/>. Training updates them in place.
        /// </summary>
        public NDArray[] Weights { get; private set; }

        /// <summary>Creates a network without layers.</summary>
        public NetworkModel()
        {
            Layers = new List<NeuralLayer>();
        }

        /// <summary>Appends a layer after the current last one. Call <see cref="Build"/> after the last layer.</summary>
        /// <param name="layer">The layer to append.</param>
        /// <exception cref="ArgumentNullException"><paramref name="layer"/> is <c>null</c>.</exception>
        public void AddLayer(NeuralLayer layer)
        {
            Layers.Add(layer ?? throw new ArgumentNullException(nameof(layer)));
        }

        /// <summary>
        ///     Creates the weight matrices for the current <see cref="Layers"/>, each weight drawn uniformly from
        ///     [0, 1). Calling it again starts training over from new random weights.
        /// </summary>
        /// <param name="seed">
        ///     The seed of the NumPy-compatible generator (<c>np.random.default_rng(seed)</c>), so a seed gives the
        ///     same weights as NumPy's <c>default_rng(seed).random((out, in))</c>, layer by layer. <c>null</c> seeds
        ///     from the operating system: every build differs.
        /// </param>
        /// <exception cref="InvalidOperationException">There are fewer than two layers, or a layer has no neurons.</exception>
        public void Build(int? seed = null)
        {
            if (Layers.Count < 2)
                throw new InvalidOperationException("A network needs at least an input and an output layer.");
            if (Layers.Any(layer => layer.Count < 1))
                throw new InvalidOperationException("Every layer needs at least one neuron.");

            var generator = seed is int value ? np.random.default_rng(value) : np.random.default_rng();
            var weights = new NDArray[Layers.Count - 1];
            for (int l = 0; l < weights.Length; l++)
                weights[l] = generator.random(new long[] { Layers[l + 1].Count, Layers[l].Count });

            // The model keeps these past this call, so no caller's NDScope may reclaim them.
            NDScope.Detach(weights);
            ReleaseWeights();
            Weights = weights;
        }

        /// <summary>
        ///     Trains the network with backpropagation: <paramref name="iterations"/> epochs, each walking the samples
        ///     in row order in batches of <paramref name="batchSize"/>.
        /// </summary>
        /// <param name="X">The samples, shape (samples, input neurons); any numeric dtype, computed as float64.</param>
        /// <param name="Y">The class index of every sample, an integer in [0, output neurons); the target is its one-hot vector.</param>
        /// <param name="iterations">The number of epochs (passes over all samples).</param>
        /// <param name="learningRate">The step size of every weight update.</param>
        /// <param name="batchSize">
        ///     The samples per update; the last batch of an epoch may be smaller. 1 is per-sample stochastic gradient
        ///     descent (the slowest on NumSharp, see the class remarks).
        /// </param>
        /// <exception cref="InvalidOperationException"><see cref="Build"/> has not been called for the current layers.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array with the input layer's width, or <paramref name="Y"/>
        ///     does not hold one class index in range per sample.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="iterations"/> is negative or <paramref name="batchSize"/> is not positive.</exception>
        public void Train(NDArray X, NDArray Y, int iterations, double learningRate = 0.1, int batchSize = 32)
        {
            if (iterations < 0)
                throw new ArgumentOutOfRangeException(nameof(iterations), iterations, "The epoch count must be 0 or more.");
            if (batchSize < 1)
                throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "The batch size must be at least 1.");

            // One scope for the inputs' conversions; every batch below opens its own (Step is [NDScoped]), so a
            // batch's activations and gradients return to the pool before the next batch allocates them again.
            using var scope = NDScope.Open();
            NDArray samples = CheckInput(X);
            NDArray targets = OneHotTargets(Y, samples.shape[0]);
            long rows = samples.shape[0];

            for (int epoch = 0; epoch < iterations; epoch++)
            {
                for (long start = 0; start < rows; start += batchSize)
                {
                    long stop = Math.Min(start + batchSize, rows);
                    Step(samples[$"{start}:{stop}"], targets[$"{start}:{stop}"], learningRate);
                }
            }
        }

        /// <summary>The output layer's activations for every sample: one forward pass of the whole batch.</summary>
        /// <param name="X">The samples, shape (samples, input neurons).</param>
        /// <returns>A float64 (samples, output neurons) array with values in (0, 1).</returns>
        /// <exception cref="InvalidOperationException"><see cref="Build"/> has not been called for the current layers.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the input layer's width.</exception>
        [NDScoped]
        public NDArray PredictProba(NDArray X)
        {
            NDArray activation = CheckInput(X);
            foreach (NDArray weights in Weights)
                activation = NDHelpers.Sigmoid(np.dot(activation, weights.T));
            return activation;
        }

        /// <summary>The class of every sample: the index of its most active output neuron.</summary>
        /// <param name="X">The samples, shape (samples, input neurons).</param>
        /// <returns>An int32 vector of class indices; a tie goes to the lower index.</returns>
        /// <exception cref="InvalidOperationException"><see cref="Build"/> has not been called for the current layers.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the input layer's width.</exception>
        [NDScoped]
        public NDArray Predict(NDArray X) => np.argmax(PredictProba(X), axis: 1).astype(np.int32);

        /// <summary>Returns the weight matrices' buffers to NumSharp's pool; <see cref="Build"/> is needed again afterwards.</summary>
        public void Dispose()
        {
            ReleaseWeights();
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///     One gradient step on one batch: the forward pass, the deltas from the output layer back, and the weight
        ///     updates. Its activations and gradients are reclaimed when it returns.
        /// </summary>
        /// <param name="x">The batch's samples, shape (batch, input neurons).</param>
        /// <param name="target">The batch's one-hot targets, shape (batch, output neurons).</param>
        /// <param name="learningRate">The step size; the batch's summed update is divided by its row count.</param>
        [NDScoped]
        private void Step(NDArray x, NDArray target, double learningRate)
        {
            int layers = Weights.Length;
            var activations = new NDArray[layers + 1];
            activations[0] = x;
            for (int l = 0; l < layers; l++)
                activations[l + 1] = NDHelpers.Sigmoid(np.dot(activations[l], Weights[l].T));

            NDArray output = activations[layers];
            NDArray delta = (target - output) * output * (1.0 - output);
            double rate = learningRate / x.shape[0];
            for (int l = layers - 1; l >= 0; l--)
            {
                NDArray input = activations[l];
                // The next delta must use this layer's weights BEFORE their update: every delta of a backward pass
                // comes from the weights the forward pass used.
                NDArray previous = l > 0 ? np.dot(delta, Weights[l]) * input * (1.0 - input) : null;
                // In place: Weights[l] was allocated outside this scope, so writing into it (instead of assigning a
                // new array, which this scope would reclaim on return) is what keeps the update.
                np.add(Weights[l], rate * np.dot(delta.T, input), @out: Weights[l]);
                delta = previous;
            }
        }

        /// <summary>Validates network input against the built layers.</summary>
        /// <param name="X">The samples.</param>
        /// <returns><paramref name="X"/> as float64 (no copy when it already is).</returns>
        /// <exception cref="InvalidOperationException"><see cref="Build"/> has not been called for the current layers.</exception>
        /// <exception cref="ArgumentException"><paramref name="X"/> is not a finite 2-D array with the input layer's width.</exception>
        [NDScopedCovered]
        private NDArray CheckInput(NDArray X)
        {
            if (Weights is null || Weights.Length != Layers.Count - 1)
                throw new InvalidOperationException("The network is not built for its current layers: call Build first.");
            NDArray samples = NDHelpers.Float64Matrix(X, nameof(X));
            if (samples.shape[1] != Layers[0].Count)
                throw new ArgumentException($"X has {samples.shape[1]} features, but the input layer has {Layers[0].Count} neurons.", nameof(X));
            return samples;
        }

        /// <summary>Turns class indices into the one-hot target rows the output layer is trained towards.</summary>
        /// <param name="Y">The class index of every sample.</param>
        /// <param name="rows">The sample count.</param>
        /// <returns>A float64 (samples, output neurons) array.</returns>
        /// <exception cref="ArgumentException"><paramref name="Y"/> is not one integer class index in range per sample.</exception>
        [NDScopedCovered]
        private NDArray OneHotTargets(NDArray Y, long rows)
        {
            NDArray labels = NDHelpers.Targets(Y, rows, nameof(Y));
            int outputs = Layers[Layers.Count - 1].Count;
            NDArray codes = labels.astype(np.int64);
            if (!(bool)np.all(codes == labels) || (bool)np.any(codes < 0) || (bool)np.any(codes >= outputs))
                throw new ArgumentException($"Y must hold integer class indices in [0, {outputs}).", nameof(Y));
            return NDHelpers.OneHot(codes, outputs);
        }

        /// <summary>Disposes the current weight matrices, if any, and forgets them.</summary>
        private void ReleaseWeights()
        {
            if (Weights is null)
                return;
            foreach (NDArray weights in Weights)
                weights.Dispose();
            Weights = null;
        }
    }
}
