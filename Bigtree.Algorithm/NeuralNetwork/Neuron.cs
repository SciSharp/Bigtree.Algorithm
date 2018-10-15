using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class Neuron
    {
        /// <summary>
        /// Input values
        /// </summary>
        public List<Dendrite> InputDendrites { get; set; }

        /// <summary>
        /// Output pulse
        /// </summary>
        public double Output { get; set; }

        public double Delta { get; set; }

        public Neuron()
        {
            InputDendrites = new List<Dendrite>();
        }

        public void Fire(NeuralLayer preLayer)
        {
            Output = Sum(preLayer);

            Output = ActivationFunction.Sigmoid(Output);

            // Console.WriteLine($"Activation: {Output}");
            // Console.WriteLine();
        }

        public void UpdateWeights(double new_weights)
        {
            foreach (var terminal in InputDendrites)
            {
                terminal.Weight = new_weights;
            }
        }

        private double Sum(NeuralLayer preLayer)
        {
            double computeValue = 0.0f;
            for(int i = 0; i < preLayer.Neurons.Count; i++)
            {
                var neuron = preLayer.Neurons[i];
                var dendrite = InputDendrites[i];

                computeValue += neuron.Output * dendrite.Weight;

                // Console.WriteLine($"{neuron.Output} * {neuron.InputDendrites[0].Weight} = {neuron.Output * neuron.InputDendrites[0].Weight}");
            }

            // Console.WriteLine($"Sum = {computeValue}");

            return computeValue;
        }
    }
}
