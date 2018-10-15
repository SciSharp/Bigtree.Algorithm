using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; set; }

        public string Name { get; set; }

        public NeuralLayer(int count, string name = "")
        {
            Neurons = new List<Neuron>();
            for (int i = 0; i < count; i++)
            {
                Neurons.Add(new Neuron());
            }

            Name = name;
        }

        public void Forward(NeuralLayer preLayer)
        {
            foreach (var neuron in Neurons)
            {
                neuron.Fire(preLayer);
            }
        }
    }
}
