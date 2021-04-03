using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using NumSharp;

namespace MNIST.IO
{
    public class ImageData
    {
        public byte Label { get; set; }
        public byte[,] Image { get; set; }

        public double[,] AsDouble()
        {
            var res = new double[Image.GetLength(0), Image.GetLength(1)];
            for (var x = 0; x<Image.GetLength(0); x++)
                for (var y = 0; y<Image.GetLength(1); y++)
                    res[x,y] = Image[x,y] / 256.0;
            return res;
        }
    }   

    /// http://yann.lecun.com/exdb/mnist/
    public class FileReaderMNIST 
    {
        public (NDArray, NDArray) LoadImagesAndLables(string labelFile, string imageFile, int total)
        {
            var labels = LoadLabel(labelFile, total);
            var images = LoadImages(imageFile, total);

            var ndImages = new NDArray(np.float64);
            var ndLabels = new NDArray(np.int32);

            var cc=0;
            foreach (var img in images)
            {
                //ndImages.Data.Add(img);
                //ndLabels.Data.Add(labels[cc]);
                cc++;
            }

            return (ndImages, ndLabels);
        }

        public IEnumerable<NDArray> LoadImages(string imageFile, int total)
        {
            /*
            TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
            [offset] [type]          [value]          [description] 
            0000     32 bit integer  0x00000803(2051) magic number 
            0004     32 bit integer  60000            number of images 
            0008     32 bit integer  28               number of rows 
            0012     32 bit integer  28               number of columns 
            0016     unsigned byte   ??               pixel 
            0017     unsigned byte   ??               pixel 
            ........ 
            xxxx     unsigned byte   ??               pixel
            Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). */
            
            using (var raw = File.OpenRead(imageFile))
            {
                using(var gz = new GZipStream(raw, CompressionMode.Decompress))
                {
                    using (var reader = new BinaryReaderMSB(gz))
                    {
                        var header = reader.ReadInt32MSB();
                        if (header != 0x803) throw new InvalidDataException(header.ToString("x"));  
                        var itemCount = reader.ReadInt32MSB();
                        var rowCount = reader.ReadInt32MSB();  
                        var colCount = reader.ReadInt32MSB();

                        for(var i=0; i< total; i++)
                        {
                            var nd = np.zeros(rowCount, colCount);

                            for(var r=0; r<rowCount; r++)
                                for(var c=0; c<colCount; c++)
                                    nd[r, c] = reader.ReadByte();

                            yield return nd;
                        }
                    }
                }
            }
        }

        public byte[] LoadLabel(string labelFile, int total)
        {
            /*
            TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
            [offset] [type]          [value]          [description] 
            0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
            0004     32 bit integer  60000            number of items 
            0008     unsigned byte   ??               label 
            0009     unsigned byte   ??               label 
            ........ 
            xxxx     unsigned byte   ??               label
            The labels values are 0 to 9.
            */
            using (var raw = File.OpenRead(labelFile))
            {
                using (var gz = new GZipStream(raw, CompressionMode.Decompress))
                {
                    using (var reader = new BinaryReaderMSB(gz))
                    {
                        // Check Header / Magic Number
                        var header = reader.ReadInt32MSB();
                        if (header != 0x801) throw new InvalidDataException(header.ToString("x"));
                        var itemCount = reader.ReadInt32MSB();
                        return reader.ReadBytes(total);
                    }
                }
            }
        }
    }
}
