using NumSharp;
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
        public static DataSet<T1, T2> ReadCsv<T1, T2>(string path) 
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

            return new DataSet<T1, T2> { X = X, Y = y };
        }
    }
}
