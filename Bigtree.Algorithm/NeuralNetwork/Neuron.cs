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
        public List<Dendrite> Dendrites { get; set; }

        /// <summary>
        /// Output pulse
        /// </summary>
        public Pulse OutputPulse { get; set; }

        private double threshold;

        public Neuron()
        {
            Dendrites = new List<Dendrite>();
            OutputPulse = new Pulse();

            threshold = 1;
        }

        public void Fire()
        {
            OutputPulse.Value = Sum();

            OutputPulse.Value = ActivationFunction.Step(OutputPulse.Value, threshold);

            Console.WriteLine($"Activation: {OutputPulse.Value}");
            Console.WriteLine();
        }

        public void UpdateWeights(double new_weights)
        {
            foreach (var terminal in Dendrites)
            {
                terminal.SynapticWeight = new_weights;
            }
        }

        private double Sum()
        {
            double computeValue = 0.0f;
            foreach (var d in Dendrites)
            {
                computeValue += d.InputPulse.Value * d.SynapticWeight;
                Console.WriteLine($"{d.InputPulse.Value} * {d.SynapticWeight} = {d.InputPulse.Value * d.SynapticWeight}");
            }

            Console.WriteLine($"Sum = {computeValue}");

            return computeValue;
        }
    }
}
