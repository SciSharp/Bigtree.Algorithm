using Bigtree.Algorithm.MathFunctions;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class NetworkModel
    {
        public List<NeuralLayer> Layers { get; set; }

        public NetworkModel()
        {
            Layers = new List<NeuralLayer>();
        }

        public void AddLayer(NeuralLayer layer)
        {
            int dendriteCount = 1;

            if (Layers.Count > 0)
            {
                dendriteCount = Layers.Last().Neurons.Count;
            }

            foreach (var element in layer.Neurons)
            {
                for (int i = 0; i < dendriteCount; i++)
                {
                    element.InputDendrites.Add(new Dendrite());
                }
            }
        }

        public void Build()
        {
            int i = 0;
            foreach (var layer in Layers)
            {
                if (i == 0)
                {
                    layer.Neurons.ForEach(x => x.InputDendrites.Add(new Dendrite
                    {
                        Weight = 1,
                        Pulse = -1
                    }));
                }
                else if (i >= Layers.Count - 1)
                {
                    break;
                }

                var nextLayer = Layers[i + 1];
                CreateNetwork(layer, nextLayer);

                i++;
            }
        }

        public void Train(NDArray X, NDArray Y, int iterations, double learningRate = 0.1)
        {
            int epoch = 1;
            //Loop till the number of iterations
            while (iterations >= epoch)
            {
                //Loop through the record
                for (int i = 0; i < X.shape[0]; i++)
                {
                    // Create target output
                    var y_target = np.zeros<int>(Layers.Last().Neurons.Count);
                    y_target[(int)Y[i]] = 1;
                    // Forward-pass training example into network (updates node output)
                    var x_input = X[i] as NDArray;
                    ForwardPropagation(x_input);
                    // Backward-pass error into network (updates node delta)
                    BackPropagation(y_target);
                    // Update network weights (using updated node delta and node output)
                    UpdateWeights(X[i] as NDArray, learningRate);
                }

                epoch++;
            }
        }

        public NDArray Predict(NDArray X)
        {
            var y_predict = np.zeros<int>(X.shape[0]);

            for (int i = 0; i < X.shape[0]; i++)
            {
                var x = X[i] as NDArray;
                ForwardPropagation(x);
                var output = Layers[Layers.Count - 1].Neurons.Select(neuron => neuron.Output).ToList();
                y_predict[i] = np.array(output.ToArray()).argmax();
            }

            return y_predict;
        }

        public NDArray Predict2(NDArray X)
        {
            var y_predict = np.zeros<int>(X.size);
            var nd = new NDArray(np.float64);

            for (int i = 0; i < X.size; i++)
            {
                var x = X[i] as NDArray;
                ForwardPropagation(x);
                var output = Layers[Layers.Count - 1].Neurons.Select(neuron => neuron.Output).ToList();
                y_predict[i] = np.array(output.ToArray()).argmax();
            }

            return y_predict;
        }

        private void BackPropagation(NDArray target)
        {
            for (int layerIndex = Layers.Count - 1; layerIndex > 0; layerIndex--)
            {
                var layer = Layers[layerIndex];
                List<double> errors = new List<double>();

                for (int neuronIndex = 0; neuronIndex < layer.Neurons.Count; neuronIndex++)
                {
                    // Last layer: errors = target output difference 
                    if (layerIndex == Layers.Count - 1)
                    {
                        var neuron = layer.Neurons[neuronIndex];
                        errors.Add((int)target[neuronIndex] - neuron.Output);
                    }
                    // Previous layers: error = weights sum of frontward node deltas
                    else
                    {
                        double error = 0;
                        for (int pre = layerIndex; pre > 0; pre--)
                        {
                            Layers[pre + 1].Neurons.ForEach(neuron =>
                            {
                                error += neuron.InputDendrites[neuronIndex].Weight * neuron.Delta;
                            });
                        }

                        errors.Add(error);
                    }
                }

                for (int neuronIndex = 0; neuronIndex < layer.Neurons.Count; neuronIndex++)
                {
                    var neuron = layer.Neurons[neuronIndex];
                    // Update delta using our errors
                    neuron.Delta = errors[neuronIndex] * Function.SigmoidDerivative(neuron.Output);
                }
            }
        }

        /// <summary>
        /// Perform forward-pass through network and update node outputs
        /// </summary>
        private void ForwardPropagation(NDArray data)
        {
            //Set the input data into the first layer
            Layers[0].Neurons.Select((x, i) => x.Output = (double)data[i]).ToList();

            NeuralLayer preLayer = null;

            foreach (var layer in Layers)
            {
                //Skip first layer as it is input
                if(preLayer == null)
                {
                    preLayer = layer;
                    continue;
                }

                layer.Forward(preLayer);

                preLayer = layer;
            }
        }

        private void UpdateWeights(NDArray x, double learningRate = 0.1)
        {
            for (int layerIndex = 1; layerIndex < Layers.Count; layerIndex++)
            {
                var inputs = x;

                var layer = Layers[layerIndex];

                if (layerIndex > 1)
                {
                    var layer2 = Layers[layerIndex - 1];
                    inputs = new NDArray(np.float64);
                    inputs.Storage.SetData(layer2.Neurons.Select(output => output.Output).ToArray());
                    inputs.reshape();
                }

                for (int n = 0; n < layer.Neurons.Count; n++)
                {
                    var neuron = layer.Neurons[n];

                    for (int i = 0; i < inputs.size; i++)
                    {
                        neuron.InputDendrites[i].Weight += learningRate * neuron.Delta * (double)x[i];
                    }
                    
                }
            }
        }

        private void CreateNetwork(NeuralLayer connectingFrom, NeuralLayer connectingTo)
        {
            // Initial a rand weight for input layer
            var rand = new Random();

            foreach (var to in connectingTo.Neurons)
            {
                int i = 0;
                to.InputDendrites = new List<Dendrite>();
                foreach (var from in connectingFrom.Neurons)
                {
                    i++;
                    to.InputDendrites.Add(new Dendrite()
                    {
                        Pulse = from.Output,
                        Weight = rand.NextDouble() // i / 10.0 + i / 100.0
                    });
                }
            }
        }
    }
}
