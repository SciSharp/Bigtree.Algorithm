using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    /// <summary>
    /// A pulse is an electric signal passing through the dendrite of neuron 
    /// which forms the basis of data (value stored in double datatype). 
    /// </summary>
    public class Pulse
    {
        public Pulse()
        {
            Value = -1;
        }

        public double Value { get; set; }
    }
}
