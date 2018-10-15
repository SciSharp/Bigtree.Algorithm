using Bigtree.Algorithm.MathFunctions;
using NumSharp;
using NumSharp.Extensions;
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

        public void Train(NdArray<NdArray<double>> X, NdArray<int> Y, int iterations, double learningRate = 0.1)
        {
            int epoch = 1;
            //Loop till the number of iterations
            while (iterations >= epoch)
            {
                //Loop through the record
                for (int i = 0; i < X.Data.Count; i++)
                {
                    FireForwardPropagation(X.Data[i]);

                    // Create target output
                    var y_target = new NdArray<int>().Zeros(Layers.Last().Neurons.Count);
                    y_target[Y[i] - 1] = 1;
                    // Backward-pass error into network (updates node delta)
                    BackPropagation(y_target);
                    // Update network weights (using updated node delta and node output)
                    UpdateWeights(X.Data[i], learningRate);
                }

                epoch++;
            }
        }

        public NdArray<int> Predict(NdArray<NdArray<double>> X)
        {
            var y_predict = new NdArray<int>().Zeros(X.Length);
            var np = new NdArray<double>();

            for (int i = 0; i < X.Length; i++)
            {
                var x = X[i];
                FireForwardPropagation(x);
                var output = Layers[Layers.Count - 1].Neurons.Select(neuron => neuron.Output).ToList();
                y_predict[i] = np.Array(output).ArgMax() + 1;
            }

            return y_predict;
        }

        /// <summary>
        /// Perform forward-pass through network and update node outputs
        /// </summary>
        private void FireForwardPropagation(NdArray<double> data)
        {
            //Get the input layers
            var inputLayer = Layers[0];

            //Set the input data into the first layer
            for (int j = 0; j < data.Length; j++)
            {
                inputLayer.Neurons[j].Output = data[j];
            }

            //Fire all the neurons and collect the output
            ComputeOutput();
        }

        private void BackPropagation(NdArray<int> target)
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
                        errors.Add(target[neuronIndex] - neuron.Output);
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

        private void ComputeOutput()
        {
            bool first = true;
            NeuralLayer preLayer = null;

            foreach (var layer in Layers)
            {
                //Skip first layer as it is input
                if (first)
                {
                    first = false;
                    preLayer = layer;
                    continue;
                }

                layer.Forward(preLayer);

                preLayer = layer;
            }
        }

        private void UpdateWeights(NdArray<double> x, double learningRate = 0.1)
        {
            for (int layerIndex = 1; layerIndex < Layers.Count; layerIndex++)
            {
                var inputs = x;

                var layer = Layers[layerIndex];

                if (layerIndex > 1)
                {
                    var layer2 = Layers[layerIndex - 1];
                    inputs = new NdArray<double>() { NDim = 1 };
                    inputs.Data = layer2.Neurons.Select(output => output.Output).ToList();
                }

                for(int n = 0; n < layer.Neurons.Count; n++)
                {
                    var neuron = layer.Neurons[n];
                    neuron.InputDendrites[n].Weight = x.Data.Sum(input => learningRate * neuron.Delta * input);
                }
            }
        }

        private void OptimizeWeights(double accuracy)
        {
            float lr = 0.1f;
            //Skip if the accuracy reached 100%
            if (accuracy == 1)
            {
                return;
            }

            if (accuracy > 1)
            {
                lr = -lr;
            }

            //Update the weights for all the layers
            foreach (var layer in Layers)
            {
                layer.Optimize(lr, 1);
            }
        }

        private void CreateNetwork(NeuralLayer connectingFrom, NeuralLayer connectingTo)
        {
            // Initial a rand weight for input layer
            var rand = new Random();

            foreach (var to in connectingTo.Neurons)
            {
                to.InputDendrites = new List<Dendrite>();
                foreach (var from in connectingFrom.Neurons)
                {
                    to.InputDendrites.Add(new Dendrite()
                    {
                        Pulse = from.Output,
                        Weight = rand.NextDouble()
                    });
                }
            }
        }
    }
}
