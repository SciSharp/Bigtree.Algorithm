using System;

namespace Bigtree.Algorithm.NeuralNetwork
{
    /// <summary>
    ///     One layer of a <see cref="NetworkModel"/>: how many neurons it has and a display name. The weights live in
    ///     the model's <see cref="NetworkModel.Weights"/> matrices, created by <see cref="NetworkModel.Build"/>.
    /// </summary>
    public class NeuralLayer
    {
        /// <summary>The number of neurons; for the input layer the feature count, for the output layer the class count.</summary>
        public int Count { get; }

        /// <summary>A display name, such as "INPUT", "HIDDEN" or "OUTPUT"; not used by the computation.</summary>
        public string Name { get; set; }

        /// <summary>Creates a layer.</summary>
        /// <param name="count">The number of neurons, at least 1.</param>
        /// <param name="name">A display name.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is less than 1.</exception>
        public NeuralLayer(int count, string name = "")
        {
            if (count < 1)
                throw new ArgumentOutOfRangeException(nameof(count), count, "A layer needs at least one neuron.");
            Count = count;
            Name = name;
        }

        /// <summary>The layer's name and size, for debugging.</summary>
        /// <returns>For example <c>"HIDDEN (5)"</c>.</returns>
        public override string ToString() => $"{Name} ({Count})";
    }
}
