using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class Dendrite
    {
        public Dendrite()
        {
            InputPulse = new Pulse();
        }

        /// <summary>
        /// Input pulse
        /// </summary>
        public Pulse InputPulse { get; set; }

        /// <summary>
        /// Synaptic weight
        /// </summary>
        public double SynapticWeight { get; set; }
    }
}
