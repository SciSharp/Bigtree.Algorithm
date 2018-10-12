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
        public double OutputPulse { get; set; }

        private double threshold;

        public Neuron()
        {
            InputDendrites = new List<Dendrite>();

            threshold = 1;
        }

        public void Fire(NeuralLayer preLayer)
        {
            OutputPulse = Sum(preLayer);

            OutputPulse = ActivationFunction.Sigmoid(OutputPulse);

            Console.WriteLine($"Activation: {OutputPulse}");
            Console.WriteLine();
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
            foreach(var neuron in preLayer.Neurons)
            {
                computeValue += neuron.OutputPulse * neuron.InputDendrites[0].Weight;
                Console.WriteLine($"{neuron.OutputPulse} * {neuron.InputDendrites[0].Weight} = {neuron.OutputPulse * neuron.InputDendrites[0].Weight}");
            }

            Console.WriteLine($"Sum = {computeValue}");

            return computeValue;
        }
    }
}
