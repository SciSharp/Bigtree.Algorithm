/*
 * SVM.NET Library
 * Copyright (C) 2008 Matthew Johnson
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


using System;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using System.Globalization;

namespace Bigtree.Algorithm.SVM
{
    /// <summary>
    /// Encapsulates a problem, or set of vectors which must be classified.
    /// </summary>
    [Serializable]
    public class Problem
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="count">Number of vectors</param>
        /// <param name="y">The class labels</param>
        /// <param name="x">Vector data.</param>
        /// <param name="maxIndex">Maximum index for a vector</param>
        public Problem(int count, double[] y, Node[][] x, int maxIndex)
        {
            Count = count;
            Y = y;
            X = x;
            MaxIndex = maxIndex;
        }
        /// <summary>
        /// Empty Constructor.  Nothing is initialized.
        /// </summary>
        public Problem()
        {
        }
        /// <summary>
        /// Number of vectors.
        /// </summary>
        public int Count { get; set; }

        /// <summary>
        /// Class labels.
        /// </summary>
        public double[] Y { get; set; }

        /// <summary>
        /// Vector data.
        /// </summary>
        public Node[][] X { get; set; }

        /// <summary>
        /// Maximum index for a vector.
        /// </summary>
        public int MaxIndex { get; set; }

        public override bool Equals(object obj)
        {
            Problem other = obj as Problem;
            if (other == null)
                return false;

            return other.Count == Count &&
                other.MaxIndex == MaxIndex &&
                other.X.IsEqual(X) &&
                other.Y.IsEqual(Y);
        }

        public override int GetHashCode()
        {
            return Count.GetHashCode() +
                MaxIndex.GetHashCode() +
                X.ComputeHashcode() +
                Y.ComputeHashcode();
        }

        /// <summary>
        ///     Builds a problem from a dense NumSharp sample matrix and its labels (or regression targets), with
        ///     libsvm's 1-based feature indices: column <c>j</c> of <paramref name="X"/> becomes feature <c>j + 1</c>.
        /// </summary>
        /// <param name="X">The samples, shape (samples, features); any numeric dtype, read as float64.</param>
        /// <param name="y">One label (classification) or target value (regression) per sample, numeric.</param>
        /// <returns>
        ///     A problem whose every sample lists all its features, zeros included, so <see cref="RangeTransform"/>
        ///     scales every feature of every sample (libsvm's sparse format would drop zeros, which scaling then never
        ///     reaches). <see cref="MaxIndex"/> is the feature count.
        /// </returns>
        /// <remarks>
        ///     One bulk copy reads the matrix out of NumSharp; the <see cref="Node"/> rows are the solver's own sparse
        ///     format. <see cref="Prediction.PredictLabels(Model, NumSharp.NDArray)"/> and
        ///     <see cref="Prediction.DecisionValues(Model, NumSharp.NDArray)"/> read query matrices with the same
        ///     column-to-index rule.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="X"/> is not a finite 2-D array, or <paramref name="y"/>'s length differs from the sample count.
        /// </exception>
        public static Problem FromNDArray(NumSharp.NDArray X, NumSharp.NDArray y)
        {
            var problem = new Problem();
            problem.Read(X, y);
            return problem;
        }

        /// <summary>
        ///     The body of <see cref="FromNDArray"/>: validates the arrays and fills this problem's
        ///     <see cref="Count"/>, <see cref="Y"/>, <see cref="X"/> and <see cref="MaxIndex"/> from them.
        /// </summary>
        /// <param name="X">The samples (see <see cref="FromNDArray"/>).</param>
        /// <param name="y">One label or target value per sample.</param>
        /// <remarks>
        ///     Every NumSharp temporary returns to the pool when the woven scope closes. <c>[NDScoped]</c> cannot sit on
        ///     <see cref="FromNDArray"/> itself: the weaver refuses a method returning a reference type that is not an
        ///     NDArray carrier (NDW003), and it returns the problem.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> or <paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">See <see cref="FromNDArray"/>.</exception>
        [NumSharp.NDScoped]
        private void Read(NumSharp.NDArray X, NumSharp.NDArray y)
        {
            NumSharp.NDArray samples = Internal.NDHelpers.Float64Matrix(X, nameof(X));
            NumSharp.NDArray targets = Internal.NDHelpers.Targets(y, samples.shape[0], nameof(y));
            int rows = (int)samples.shape[0], features = (int)samples.shape[1];

            // ToArray copies the (possibly strided) matrix out in row-major order in one pass.
            double[] values = samples.ToArray<double>();
            var nodes = new Node[rows][];
            for (int i = 0; i < rows; i++)
            {
                var row = new Node[features];
                for (int j = 0; j < features; j++)
                    row[j] = new Node(j + 1, values[(long)i * features + j]);
                nodes[i] = row;
            }

            // this. because the parameters X and y shadow the properties X (and read like Y).
            this.Count = rows;
            this.Y = targets.astype(NumSharp.np.float64).ToArray<double>();
            this.X = nodes;
            this.MaxIndex = features;
        }

        /// <summary>
        /// Reads a problem from a stream.
        /// </summary>
        /// <param name="stream">Stream to read from</param>
        /// <returns>The problem</returns>
        public static Problem Read(Stream stream)
        {
            TemporaryCulture.Start();

            StreamReader input = new StreamReader(stream);
            List<double> vy = new List<double>();
            List<Node[]> vx = new List<Node[]>();
            int max_index = 0;

            while (input.Peek() > -1)
            {
                string[] parts = input.ReadLine().Trim().Split();

                vy.Add(double.Parse(parts[0]));
                int m = parts.Length - 1;
                Node[] x = new Node[m];
                for (int j = 0; j < m; j++)
                {
                    x[j] = new Node();
                    string[] nodeParts = parts[j + 1].Split(':');
                    x[j].Index = int.Parse(nodeParts[0]);
                    x[j].Value = double.Parse(nodeParts[1]);
                }
                if (m > 0)
                    max_index = Math.Max(max_index, x[m - 1].Index);
                vx.Add(x);
            }

            TemporaryCulture.Stop();

            return new Problem(vy.Count, vy.ToArray(), vx.ToArray(), max_index);
        }

        /// <summary>
        /// Writes a problem to a stream.
        /// </summary>
        /// <param name="stream">The stream to write the problem to.</param>
        /// <param name="problem">The problem to write.</param>
        public static void Write(Stream stream, Problem problem)
        {
            TemporaryCulture.Start();

            StreamWriter output = new StreamWriter(stream);
            for (int i = 0; i < problem.Count; i++)
            {
                output.Write(problem.Y[i]);
                for (int j = 0; j < problem.X[i].Length; j++)
                    output.Write(" {0}:{1:0.000000}", problem.X[i][j].Index, problem.X[i][j].Value);
                output.Write("\n");
            }
            output.Flush();

            TemporaryCulture.Stop();
        }

        /// <summary>
        /// Reads a Problem from a file.
        /// </summary>
        /// <param name="filename">The file to read from.</param>
        /// <returns>the Probem</returns>
        public static Problem Read(string filename)
        {
            FileStream input = File.OpenRead(filename);
            try
            {
                return Read(input);
            }
            finally
            {
                input.Close();
            }
        }

        /// <summary>
        /// Writes a problem to a file.   This will overwrite any previous data in the file.
        /// </summary>
        /// <param name="filename">The file to write to</param>
        /// <param name="problem">The problem to write</param>
        public static void Write(string filename, Problem problem)
        {
            FileStream output = File.Open(filename, FileMode.Create);
            try
            {
                Write(output, problem);
            }
            finally
            {
                output.Close();
            }
        }
    }
}