using System;
using System.IO;

namespace MNIST.IO
{
    public class BinaryReaderMSB : BinaryReader
    {
        public BinaryReaderMSB(Stream stream) : base(stream)
        {
            
        }

        public int ReadInt32MSB()
        {
            var s = ReadBytes(4);
            Array.Reverse(s);
            return BitConverter.ToInt32(s, 0);
        }
    }
}
