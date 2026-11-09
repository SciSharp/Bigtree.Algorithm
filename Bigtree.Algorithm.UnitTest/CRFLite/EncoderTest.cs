using Bigtree.Algorithm.CRFLite;
using Bigtree.Algorithm.CRFLite.Encoder;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;

namespace Bigtree.Algorithm.UnitTest.CRFLite
{
    /// <summary>
    ///     Training a CRF model. Bigtree 0.4.0 read its corpus and template from its author's disk
    ///     (<c>C:\Users\haipi\...</c>); the test now trains on a small named-entity corpus that ships with the tests
    ///     (<c>data/crf</c>: word, part of speech, tag; tab-separated, a blank line between sentences).
    /// </summary>
    [TestClass]
    public class EncoderTest : TestEssential
    {
        /// <summary>
        ///     Trains a model from the corpus and the template into a temporary folder and checks the model files exist.
        /// </summary>
        [TestMethod]
        public void TestEncode()
        {
            string folder = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "bigtree-crf-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            try
            {
                string model = System.IO.Path.Combine(folder, "ner_model");
                var encoder = new CRFEncoder();
                bool result = encoder.Learn(new EncoderOptions
                {
                    /*
                     * traing corups format, split by tab, sentences is seperated by blank row
                     *
                        Tokyo NNP S_LOCATION
                        and	CC S
                        New	NNP	B_LOCATION
                        York NNP	E_LOCATION
                        are	VBP	S
                        major JJ S
                        financial JJ S
                        centers	NNS	S
                        . PUN S
                     */
                    TrainingCorpusFileName = System.IO.Path.Combine(DataDir, "crf", "train.txt"),
                    TemplateFileName = System.IO.Path.Combine(DataDir, "crf", "template.txt"),
                    ModelFileName = model,
                    MaxIteration = 60,
                });

                Assert.IsTrue(result);
                Assert.IsTrue(File.Exists(model), "the model's metadata file");
                Assert.IsTrue(new FileInfo(model).Length > 0);
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }
}
