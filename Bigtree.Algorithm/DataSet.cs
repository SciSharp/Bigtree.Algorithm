using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm
{
    public class DataSet<T1, T2>
    {
        public NdArray<NdArray<T1>> X { get; set; }
        public NdArray<T2> Y { get; set; }
    }
}
