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
        public static (NDArray<double>, NDArray<int>) ReadCsv(string path) 
        {
            List<int> labels = new List<int>();

            var x1 = new List<double>();
            var y1 = new List<int>();
            int length1d = 0;
            int length2d = 0;

            using (StreamReader reader = new StreamReader(path))
            {
                string line = String.Empty;

                while (!String.IsNullOrEmpty(line = reader.ReadLine()))
                {
                    var tokens = line.Split(',');
                    x1.AddRange(tokens.Select(x => double.Parse(x)).Take(tokens.Length - 1));

                    var _y = int.Parse(tokens[tokens.Length - 1]);
                    if (!labels.Contains(_y))
                    {
                        labels.Add(_y);
                    }
                    y1.Add(labels.FindIndex(l => l == _y));

                    length1d++;
                    length2d = tokens.Length - 1;
                }
            }

            var X = new NDArray<double>().Array(x1.ToArray()).ReShape(length1d, length2d);
            var y = new NDArray<int>().Array(y1.ToArray());

            return (X, y);
        }

        public static List<List<int>> CrossValFolds(int N, int n_folds)
        {
            var folds = new List<List<int>>();

            var np = new NumPy<int>();
            var rands = np.random.Permutation(N);

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
