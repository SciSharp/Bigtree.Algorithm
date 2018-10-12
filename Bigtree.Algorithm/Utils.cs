using NumSharp;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;

namespace Bigtree.Algorithm
{
    public class Utils
    {
        public static (NdArray<NdArray<T1>>, NdArray<T2>) ReadCsv<T1, T2>(string path) 
        {
            var X = new NdArray<NdArray<T1>>
            {
                Data = new List<NdArray<T1>>()
            };

            var y = new NdArray<T2>()
            {
                NDim = 1,
                Data = new List<T2>()
            };

            using (StreamReader reader = new StreamReader(path))
            {
                string line = String.Empty;

                while (!String.IsNullOrEmpty(line = reader.ReadLine()))
                {
                    var tokens = line.Split(',');

                    if (X.NDim == -1)
                    {
                        X.NDim = tokens.Length - 1;
                    }

                    NdArray<T1> row = new NdArray<T1>
                    {
                        NDim = 1
                    };

                    row.Data = tokens.Take(X.NDim).Select(x => (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFrom(x)).ToList();

                    X.Data.Add(row);
                    y.Data.Add((T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFrom(tokens[tokens.Length - 1]));
                }
            }

            return (X, y);
        }

        public static List<List<int>> CrossValFolds(int N, int n_folds)
        {
            var folds = new List<List<int>>();

            var np = new NdArray<int>();
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
