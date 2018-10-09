/*
 * Bigtree.Algorithm
 * Copyright (C) 2018 Haiping Chen
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Apache License 2.0 as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the Apache License 2.0
 * along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.0/>.
 */

using Bigtree.Algorithm.MathFunctions;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.NeuralNetwork
{
    /// <summary>
    /// In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs.
    /// </summary>
    public class ActivationFunction
    {
        /// <summary>
        /// Range: {0, 1}
        /// Monotonic: True
        /// </summary>
        /// <param name="x"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public static double Step(double x, double threshold = 1)
        {
            return x >= threshold ? threshold : 0;
        }

        /// <summary>
        /// Range: {-infinit, +infinit}
        /// Monotonic: True
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Identity(double x)
        {
            return x;
        }

        /// <summary>
        /// Rectified linear unit
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double ReLU(double x)
        {
            return x >= 0 ? x : 0;
        }

        public static double LeakyReLU(double x, double coefficient = 0.01)
        {
            return x >= 0 ? x : coefficient * x;
        }

        /// <summary>
        /// Parameteric Rectified Linear Unit
        /// </summary>
        /// <param name="x"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static double PReLU(double x, double alpha = 0.01)
        {
            throw new NotImplementedException();
            // return x >= 0 ? x : alpha * x;
        }

        /// <summary>
        /// Randomized Leaky Rectified Linear Unit
        /// </summary>
        /// <param name="x"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static double RReLU(double x, double alpha = 0.01)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Exponential Linear Unit
        /// </summary>
        /// <param name="x"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static double ELU(double x, double alpha = 0.01)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Scaled Exponential Linear Unit
        /// </summary>
        /// <param name="x"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static double SELU(double x, double alpha = 0.01)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// S-shaped Rectified Linear Activation Unit
        /// </summary>
        /// <param name="x"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public static double SReLU(double x, double alpha = 0.01)
        {
            throw new NotImplementedException();
        }

        public static double Sigmoid(double x, double coefficient = 1)
        {
            return Function.Sigmoid(x, coefficient);
        }

        /// <summary>
        /// Range: [0, 1]
        /// </summary>
        /// <param name="x"></param>
        /// <param name="coefficient"></param>
        /// <returns></returns>
        public static double HardSigmoid(double x, double coefficient = 1)
        {
            throw new NotImplementedException();
        }

        public static double SymmetricalSigmoid(double x, double coefficient = 1)
        {
            throw new NotImplementedException();
        }

        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        public static double HardTanh(double x)
        {
            throw new NotImplementedException();
        }

        public static double LeCunTanh(double x)
        {
            throw new NotImplementedException();
        }

        public static double ArcTan(double x)
        {
            throw new NotImplementedException();
        }

        public static double SoftSign(double x)
        {
            throw new NotImplementedException();
        }

        public static double SoftPlus(double x)
        {
            throw new NotImplementedException();
        }

        public static double Signum(double x)
        {
            throw new NotImplementedException();
        }

        public static double BentIdentity(double x)
        {
            throw new NotImplementedException();
        }

        public static double LogLog(double x)
        {
            throw new NotImplementedException();
        }

        public static double Gaussian(double x)
        {
            throw new NotImplementedException();
        }

        public static double Absolute(double x)
        {
            return Math.Abs(x);
        }

        public static double Sinusoid(double x)
        {
            return Math.Sin(x);
        }

        public static double Cosine(double x)
        {
            return Math.Cos(x);
        }

        /// <summary>
        /// Cardinal Sine
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Sinc(double x)
        {
            return x == 0 ? 1 : Math.Sin(x) / x;
        }
    }
}
