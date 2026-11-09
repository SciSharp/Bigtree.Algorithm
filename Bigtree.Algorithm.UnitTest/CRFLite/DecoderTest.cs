using Bigtree.Algorithm.CRFLite;
using Bigtree.Algorithm.CRFLite.Decoder;
using Bigtree.Algorithm.CRFLite.Encoder;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Bigtree.Algorithm.UnitTest.CRFLite
{
    /// <summary>
    ///     Tagging with a trained CRF model: the model is trained once from the corpus in <c>data/crf</c> (0.4.0 loaded a
    ///     model from its author's disk), then sentences are decoded and their tags checked.
    /// </summary>
    [TestClass]
    public class DecoderTest : TestEssential
    {
        /// <summary>The temporary folder of the trained model.</summary>
        private static string folder;

        /// <summary>The trained model's file.</summary>
        private static string modelFile;

        /// <summary>Trains the model the tests decode with.</summary>
        /// <param name="context">The MSTest context (unused).</param>
        /// <exception cref="AssertFailedException">Training failed.</exception>
        [ClassInitialize]
        public static void TrainModel(TestContext context)
        {
            folder = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "bigtree-crf-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            modelFile = System.IO.Path.Combine(folder, "ner_model");
            bool trained = new CRFEncoder().Learn(new EncoderOptions
            {
                TrainingCorpusFileName = System.IO.Path.Combine(DataDir, "crf", "train.txt"),
                TemplateFileName = System.IO.Path.Combine(DataDir, "crf", "template.txt"),
                ModelFileName = modelFile,
                MaxIteration = 60,
            });
            Assert.IsTrue(trained, "the CRF model trains");
        }

        /// <summary>Deletes the trained model.</summary>
        [ClassCleanup]
        public static void DeleteModel()
        {
            if (folder != null && Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }

        /// <summary>A training sentence decodes to its own tags, multi-word locations included.</summary>
        [TestMethod]
        public void TestDecode()
        {
            string[] tags = Decode(("Tokyo", "NNP"), ("and", "CC"), ("New", "NNP"), ("York", "NNP"), ("are", "VBP"),
                                   ("major", "JJ"), ("financial", "JJ"), ("centers", "NNS"), (".", "PUN"));

            CollectionAssert.AreEqual(new[] { "S_LOCATION", "S", "B_LOCATION", "E_LOCATION", "S", "S", "S", "S", "S" }, tags);
        }

        /// <summary>A sentence the model never saw, made of known words in a new order, still tags its entities.</summary>
        [TestMethod]
        public void DecodesANewSentence()
        {
            string[] tags = Decode(("Anna", "NNP"), ("visited", "VBD"), ("Madrid", "NNP"), ("and", "CC"),
                                   ("Los", "NNP"), ("Angeles", "NNP"), (".", "PUN"));

            Assert.AreEqual("S_LOCATION", tags[2]);
            Assert.AreEqual("B_LOCATION", tags[4]);
            Assert.AreEqual("E_LOCATION", tags[5]);
            Assert.AreEqual("S", tags[6]);
        }

        /// <summary>Decodes one sentence with the trained model and returns its best tag sequence.</summary>
        /// <param name="tokens">The sentence: a word and its part of speech per token.</param>
        /// <returns>One tag per token.</returns>
        private static string[] Decode(params (string word, string pos)[] tokens)
        {
            var options = new DecoderOptions { ModelFileName = modelFile, NBest = 1 };
            var decoder = new CRFDecoder();
            decoder.LoadModel(options.ModelFileName);
            var tagger = decoder.CreateTagger(options.NBest, options.MaxWord);
            tagger.set_vlevel(options.ProbLevel);

            var output = new CRFTermOut[options.NBest];
            for (var i = 0; i < options.NBest; i++)
                output[i] = new CRFTermOut(options.MaxWord);

            var dataset = tokens.Select(token => new List<string> { token.word, token.pos }).ToList();
            Assert.AreEqual(0, decoder.Segment(output, tagger, dataset), "Segment succeeds");
            return output[0].result_.Take(tokens.Length).ToArray();
        }
    }
}
