using NumSharp;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace Bigtree.Algorithm
{
    public class Utils
    {
        public static (NDArray<NDArray<double>>, NDArray<int>) ReadCsv(string path) 
        {
            List<int> labels = new List<int>();

            var X = new NDArray<NDArray<double>>
            {
                Data = new List<NDArray<double>>()
            };

            var y = new NDArray<int>();

            using (StreamReader reader = new StreamReader(path))
            {
                string line = String.Empty;

                while (!String.IsNullOrEmpty(line = reader.ReadLine()))
                {
                    var tokens = line.Split(',');

                    var row = new NDArray<double>();

                    row.Data = tokens.Take(X.NDim).Select(x => (double)TypeDescriptor.GetConverter(typeof(double)).ConvertFrom(x)).ToList();

                    X.Data.Add(row);

                    var _y = int.Parse(tokens[tokens.Length - 1]);
                    if (!labels.Contains(_y))
                    {
                        labels.Add(_y);
                    }
                    y.Data.Add(labels.FindIndex(l => l == _y));
                }
            }

            return (X, y);
        }

        public static List<List<int>> CrossValFolds(int N, int n_folds)
        {
            var folds = new List<List<int>>();

            var np = new NDArray<int>();
            var rands = np.Random().Permutation(N);

            var N_fold = N / n_folds;

            for(int i = 0; i < n_folds; i++)
            {
                var list = rands.Data.Skip(N_fold * i).Take(N_fold).ToList();
                folds.Add(list);
            }

            return folds;
        }
    }
}
