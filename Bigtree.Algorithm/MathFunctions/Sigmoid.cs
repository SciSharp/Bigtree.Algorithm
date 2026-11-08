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

using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.MathFunctions
{
    public partial class Function 
    {
        /// <summary>
        /// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="coefficient"></param>
        /// <returns></returns>
        public static double Sigmoid(double x, double coefficient = 1)
        {
            return 1 / (1 + Math.Exp(-x * coefficient));
        }

        public static double SigmoidDerivative(double y)
        {
            return y * (1 - y);
        }

        /// <summary>
        ///     The sigmoid of every element: <c>1 / (1 + exp(-x * coefficient))</c>, one NumSharp pass over the array.
        /// </summary>
        /// <param name="x">The inputs; any numeric dtype (integers compute as float64).</param>
        /// <param name="coefficient">The steepness: the slope at 0 is <c>coefficient / 4</c>.</param>
        /// <returns>A new array of <paramref name="x"/>'s shape with values in (0, 1); the caller owns it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is <c>null</c>.</exception>
        [NumSharp.NDScoped]
        public static NumSharp.NDArray Sigmoid(NumSharp.NDArray x, double coefficient = 1)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));
            return 1.0 / (1.0 + NumSharp.np.exp(-x * coefficient));
        }

        /// <summary>
        ///     The sigmoid's derivative from its outputs, element-wise: <c>y * (1 - y)</c>, where <c>y</c> already is
        ///     <c>Sigmoid(x)</c> (backpropagation keeps the outputs, so it never needs <c>x</c> again).
        /// </summary>
        /// <param name="y">Sigmoid outputs.</param>
        /// <returns>A new array of <paramref name="y"/>'s shape; the caller owns it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        [NumSharp.NDScoped]
        public static NumSharp.NDArray SigmoidDerivative(NumSharp.NDArray y)
        {
            if (y is null)
                throw new ArgumentNullException(nameof(y));
            return y * (1.0 - y);
        }
    }
}
