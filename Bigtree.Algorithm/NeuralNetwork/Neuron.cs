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
        public Pulse OutputPulse { get; set; }

        private double threshold;

        public Neuron()
        {
            InputDendrites = new List<Dendrite>();
            OutputPulse = new Pulse();

            threshold = 1;
        }

        public void Fire(NeuralLayer preLayer)
        {
            OutputPulse.Value = Sum(preLayer);

            OutputPulse.Value = ActivationFunction.Step(OutputPulse.Value);

            Console.WriteLine($"Activation: {OutputPulse.Value}");
            Console.WriteLine();
        }

        public void UpdateWeights(double new_weights)
        {
            foreach (var terminal in InputDendrites)
            {
                terminal.SynapticWeight = new_weights;
            }
        }

        private double Sum(NeuralLayer preLayer)
        {
            double computeValue = 0.0f;
            foreach(var neuron in preLayer.Neurons)
            {
                computeValue += neuron.OutputPulse.Value * neuron.InputDendrites[0].SynapticWeight;
                Console.WriteLine($"{neuron.OutputPulse.Value} * {neuron.InputDendrites[0].SynapticWeight} = {neuron.OutputPulse.Value * neuron.InputDendrites[0].SynapticWeight}");
            }

            /*foreach (var d in InputDendrites)
            {
                computeValue += d.InputPulse.Value * d.SynapticWeight;
                Console.WriteLine($"{d.InputPulse.Value} * {d.SynapticWeight} = {d.InputPulse.Value * d.SynapticWeight}");
            }*/

            Console.WriteLine($"Sum = {computeValue}");

            return computeValue;
        }
    }
}
