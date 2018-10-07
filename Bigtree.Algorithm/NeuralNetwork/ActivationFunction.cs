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
    public class ActivationFunction
    {
        public static double Step(double x, double threshold)
        {
            return x > threshold ? threshold : 0;
        }

        public static double Sigmoid(double x, double coeficient = 1)
        {
            return Function.Sigmoid(x, coeficient);
        }

        public double Rectifier(double input)
        {
            return Math.Max(0, input);
        }
    }
}
