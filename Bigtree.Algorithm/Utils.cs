using NumSharp;
using NumSharp.Core;
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
        public static (NDArray, NDArray) ReadCsv(string path) 
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

            var np = new NumPy();

            var X = np.array(x1.ToArray()).reshape(length1d, length2d);
            var y = np.array(y1.ToArray());

            return (X, y);
        }

        public static List<List<int>> CrossValFolds(int N, int n_folds)
        {
            var folds = new List<List<int>>();

            var np = new NumPy();
            var rands = np.random.permutation(N);

            var N_fold = N / n_folds;

            for(int i = 0; i < n_folds; i++)
            {
                var list = rands.Data<int>().Skip(N_fold * i).Take(N_fold).ToList();
                folds.Add(list);
            }

            return folds;
        }
    }
}
