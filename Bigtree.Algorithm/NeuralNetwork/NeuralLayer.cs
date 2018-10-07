using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; set; }

        public string Name { get; set; }

        public double Weight { get; set; }

        public NeuralLayer(int count, double initialWeight, string name = "")
        {
            Neurons = new List<Neuron>();
            for (int i = 0; i < count; i++)
            {
                Neurons.Add(new Neuron());
            }

            Weight = initialWeight;

            Name = name;
        }

        public void Optimize(double learningRate, double delta)
        {
            Weight += learningRate * delta;
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(Weight);
            }
        }

        public void Forward()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Fire();
            }
        }
    }
}
