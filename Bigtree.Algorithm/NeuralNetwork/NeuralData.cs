using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    public class NeuralData
    {
        public double[][] Data { get; set; }

        int counter = 0;

        public NeuralData(int rows)
        {
            Data = new double[rows][];
        }

        public void Add(params double[] rec)
        {
            Data[counter] = rec;
            counter++;
        }
    }
}
