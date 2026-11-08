/*
 * Bigtree.Algorithm
 * Copyright (C) 2018 Haiping Chen
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

using Bigtree.Algorithm.Statistics;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Bigtree.Algorithm.Bayes
{
    /// <summary>
    ///     Naive Bayes over labelled feature vectors: the prior of a label, the smoothed probability that feature
    ///     <c>x</c> takes a given value within a label, and the posterior that combines them.
    ///     https://en.wikipedia.org/wiki/Bayes%27_theorem
    /// </summary>
    /// <remarks>
    ///     <para><b>How it computes.</b> The conditional probabilities come from one count table per feature value:
    ///     <c>counts[label, x] = onehot(labels).T @ (X == value)</c>, a single NumSharp matrix product over the whole
    ///     <see cref="FeatureSet"/>, built the first time <see cref="CalCondProb"/> asks for that value and reused for
    ///     every later call. Bigtree 0.4.0 filtered the feature set and copied the label's rows into a fresh matrix on
    ///     every call, which made a full training pass (every label × feature × value) cost O(labels · rows · features²);
    ///     it now costs O(labels · rows · features) once, and each call is a table lookup. The numbers are the same to
    ///     the last bit: the counts are exact integers in float64 and the smoothing formula is evaluated in the same
    ///     order.</para>
    ///     <para><b>Footgun: the tables follow the list, not its contents.</b> They are rebuilt when
    ///     <see cref="FeatureSet"/> is assigned another list or its <c>Count</c> changes. Editing a row's vector (or
    ///     swapping one row for another) in place after the first <see cref="CalCondProb"/> call is not seen: assign
    ///     a new list instead.</para>
    ///     <para>Safe to call from several threads at once as long as nobody modifies <see cref="FeatureSet"/>: the
    ///     tables are built under a lock.</para>
    /// </remarks>
    public class MultinomiaNaiveBayes
    {
        /// <summary>
        ///     The labels and how often each occurs. Read by <see cref="CalPriorProb"/> only; the conditional
        ///     probabilities take their label counts from <see cref="FeatureSet"/>.
        /// </summary>
        public List<Probability> LabelDist { get; set; }

        /// <summary>
        ///     The training samples: each item is a label and the sample's feature vector. Every vector must have the
        ///     same length (the feature count).
        /// </summary>
        public List<Tuple<string, double[]>> FeatureSet { get; set; }

        /// <summary>
        ///     The additive (Lidstone) smoothing added to every count by <see cref="CalPriorProb"/> and
        ///     <see cref="CalCondProb"/>. Fixed at construction: the cached tables assume it never changes.
        /// </summary>
        private double alpha { get; set; }

        /// <summary>Guards the count tables below, which <see cref="CalCondProb"/> builds lazily.</summary>
        private readonly object _sync = new object();

        /// <summary>The <see cref="FeatureSet"/> instance the tables were counted from; <c>null</c> before the first count.</summary>
        private List<Tuple<string, double[]>> _countedSet;

        /// <summary>The <see cref="FeatureSet"/> row count when the tables were counted, to notice an Add or Remove.</summary>
        private int _countedRows;

        /// <summary>Label to row of the tables (labels in ascending ordinal order, as <c>np.unique</c> sorts text).</summary>
        private Dictionary<string, int> _labelRow;

        /// <summary>The number of samples of each label, indexed by table row.</summary>
        private double[] _labelRows;

        /// <summary>The label index (table row) of every sample of the counted set, in <see cref="FeatureSet"/> order.</summary>
        private long[] _codes;

        /// <summary>The feature count shared by every vector of the counted set.</summary>
        private int _features;

        /// <summary>
        ///     Per feature value: the log-probabilities <c>log((n[label, x] + alpha) / (n[label] + n[label] * alpha))</c>,
        ///     row-major (label, feature).
        /// </summary>
        private readonly Dictionary<double, double[]> _logCondByValue = new Dictionary<double, double[]>();

        /// <summary>Creates the classifier.</summary>
        /// <param name="alpha">
        ///     The additive smoothing added to every count (Lidstone; 1 = Laplace). With 0 a value never seen within a
        ///     label has probability 0 and log-probability negative infinity.
        /// </param>
        public MultinomiaNaiveBayes(double alpha = 0.5)
        {
            this.alpha = alpha;
        }

        /// <summary>
        ///     The smoothed prior probability of a label: <c>(N_y + alpha) / (N + k * alpha)</c>, with <c>N_y</c> the
        ///     label's frequency in <see cref="LabelDist"/>, <c>N</c> the number of samples in <see cref="FeatureSet"/>
        ///     and <c>k</c> the number of labels in <see cref="LabelDist"/>.
        /// </summary>
        /// <param name="Y">The label.</param>
        /// <returns>The prior probability of <paramref name="Y"/>.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="Y"/> is not in <see cref="LabelDist"/>.</exception>
        public double CalPriorProb(string Y)
        {
            int N = FeatureSet.Count;
            int k = LabelDist.Count;
            int Nyk = LabelDist.First(x => x.Value == Y).Freq;

            return (Nyk + alpha) / (N + k * alpha);
        }

        /// <summary>
        ///     The natural log of the smoothed probability that feature <paramref name="x"/> equals
        ///     <paramref name="feature"/> among the samples labelled <paramref name="Y"/>:
        ///     <c>log((n_yx + alpha) / (n_y + n_y * alpha))</c>, where <c>n_y</c> counts the samples labelled
        ///     <paramref name="Y"/> and <c>n_yx</c> those of them whose feature <paramref name="x"/> equals
        ///     <paramref name="feature"/> exactly.
        /// </summary>
        /// <param name="x">The feature index, 0-based.</param>
        /// <param name="Y">The label; it must occur in <see cref="FeatureSet"/>.</param>
        /// <param name="feature">The feature value. A value that never occurs counts 0.</param>
        /// <returns>The log-probability (bit-identical to Bigtree 0.4.0's, which rebuilt the label's matrix per call).</returns>
        /// <remarks>
        ///     The first call for a given <paramref name="feature"/> value counts the whole <see cref="FeatureSet"/>
        ///     with one matrix product (see the class remarks); later calls for that value are a table lookup.
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="Y"/> labels no sample, or <paramref name="x"/> is not a feature index.
        /// </exception>
        /// <exception cref="ArgumentException">The vectors of <see cref="FeatureSet"/> differ in length.</exception>
        public double CalCondProb(int x, string Y, double feature)
        {
            double[] table;
            int row;
            lock (_sync)
            {
                CountLabels();
                if (!_labelRow.TryGetValue(Y, out row))
                    throw new ArgumentOutOfRangeException(nameof(Y), Y, "No sample in FeatureSet has this label.");
                if (x < 0 || x >= _features)
                    throw new ArgumentOutOfRangeException(nameof(x), x, $"The feature vectors have {_features} features.");
                if (!_logCondByValue.TryGetValue(feature, out table))
                    _logCondByValue[feature] = table = CountValue(feature);
            }

            return table[row * _features + x];
        }

        /// <summary>
        /// calculate posterior probability P(Y|X)
        /// X is feature set, Y is label
        /// P(X1,...,Xn|Y) = Sum(P(X1|Y) +...+ P(Xn|Y)
        /// P(X, Y) = P(Y|X)P(X) = P(X|Y)P(Y) => P(Y|X) = P(Y)P(X|Y)/P(X)
        /// </summary>
        /// <param name="Y">The label.</param>
        /// <param name="features">The sample's feature vector; only the features equal to 1 contribute.</param>
        /// <param name="priorProb">The label's prior probability (see <see cref="CalPriorProb"/>).</param>
        /// <param name="condProbDictionary">
        ///     The conditional log-probabilities, keyed <c>"{Y} f{x} {value}"</c> as CherubNLP builds them from
        ///     <see cref="CalCondProb"/>.
        /// </param>
        /// <returns><c>2^(log(priorProb) + sum of the log-probabilities of the features equal to 1)</c>.</returns>
        /// <exception cref="KeyNotFoundException">A feature equal to 1 has no entry in <paramref name="condProbDictionary"/>.</exception>
        public double CalPosteriorProb(string Y, double[] features, double priorProb, Dictionary<string, double> condProbDictionary)
        {
            int featureCount = features.Length;

            double postProb = Math.Log(priorProb);

            // loop features
            for (int x = 0; x < featureCount; x++)
            {
                // Only a feature equal to 1 reads the dictionary, so build its key only then: 0.4.0 formatted a key
                // for every feature first, which made this loop allocate one string per feature per label.
                if(features[x] == 1)
                {
                    string key = $"{Y} f{x} {features[x]}";
                    postProb += condProbDictionary[key];
                }
            }

            return Math.Pow(2, postProb);
        }

        /// <summary>
        ///     (Re)counts the labels of <see cref="FeatureSet"/> when it is another list than the one counted, or its
        ///     row count changed, and drops the value tables of the previous set. Runs under <see cref="_sync"/>.
        /// </summary>
        /// <exception cref="ArgumentException">The set is empty, or its vectors differ in length.</exception>
        [NDScoped]
        private void CountLabels()
        {
            if (ReferenceEquals(_countedSet, FeatureSet) && _countedRows == FeatureSet.Count)
                return;

            if (FeatureSet.Count == 0)
                throw new ArgumentException("FeatureSet holds no samples.", nameof(FeatureSet));
            int features = FeatureSet[0].Item2.Length;
            if (FeatureSet.Any(sample => sample.Item2.Length != features))
                throw new ArgumentException($"Every feature vector must have {features} features, as the first one has.", nameof(FeatureSet));

            // np.unique sorts the text labels and numbers each sample's label by its rank: the table rows.
            var labels = np.unique(np.array(FeatureSet.Select(sample => sample.Item1).ToArray()), return_inverse: true, return_counts: true);
            string[] names = Internal.NDHelpers.Strings(labels.values);
            _labelRow = new Dictionary<string, int>(names.Length);
            for (int i = 0; i < names.Length; i++)
                _labelRow[names[i]] = i;
            _labelRows = labels.counts.astype(np.float64).ToArray<double>();
            _codes = labels.inverse_indices.ToArray<long>();
            _features = features;
            _logCondByValue.Clear();
            _countedSet = FeatureSet;
            _countedRows = FeatureSet.Count;
        }

        /// <summary>
        ///     Counts, for every label and feature at once, the samples whose feature equals <paramref name="feature"/>,
        ///     and turns the counts into smoothed log-probabilities. Runs under <see cref="_sync"/>, after
        ///     <see cref="CountLabels"/>.
        /// </summary>
        /// <param name="feature">The feature value to count.</param>
        /// <returns>The (labels, features) log-probabilities, row-major.</returns>
        private double[] CountValue(double feature)
        {
            // Scoped by hand: every array below is a temporary, and the weaver refuses [NDScoped] on a method that
            // returns a managed array (NDW003), even one of doubles that cannot hold an NDArray.
            using var scope = NDScope.Open();
            int samples = _countedSet.Count, labels = _labelRows.Length;
            var flat = new double[(long)samples * _features];
            for (int i = 0; i < samples; i++)
                Array.Copy(_countedSet[i].Item2, 0, flat, (long)i * _features, _features);

            NDArray X = np.array(flat).reshape(samples, _features);
            NDArray memberOf = (np.arange(labels).reshape(labels, 1) == np.array(_codes).reshape(1, samples)).astype(np.float64);
            // (labels, samples) @ (samples, features): row y, column x = how many samples labelled y have X[:, x] == feature.
            // Integer counts below 2^53 are exact in float64, whatever order the product sums them in.
            NDArray hits = np.dot(memberOf, (X == feature).astype(np.float64));
            NDArray n = np.array(_labelRows).reshape(labels, 1);
            // The same operations, in the same order, as 0.4.0's Math.Log((Nykx + alpha) / (Nyk + n * alpha)).
            NDArray logCond = np.log((hits + alpha) / (n + n * alpha));
            return logCond.ToArray<double>();
        }
    }
}
