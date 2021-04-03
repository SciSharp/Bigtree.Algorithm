using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.Search
{
    [TestClass]
    public class FindDuplicatedNumTest
    {
        private NDArray nd;
        private int repeated;

        [TestInitialize]
        public void Initialize()
        {
            repeated = -1;
            nd = np.arange(10);
            nd[8] = 5;
            np.random.shuffle(nd);
        }

        [TestMethod]
        public void FindDuplicatedNumber()
        {
            repeated = Method1(nd.ToArray<int>());

            repeated = Method2(nd.ToArray<int>());

            repeated = Method4(nd.ToArray<int>());
        }

        private int Method1(int[] a)
        {
            int i;
            int end = nd.size - 1;

            for (i = 0; i <= end; i++)
                a[end] += a[i];
            a[end] -= i * (i - 1) / 2;

            return a[end];
        }

        private int Method2(int[] array)
        {
            int index = 0;
            while (true)
            {
                if (array[index] < 0)
                    break;
                array[index] *= -1;
                index = array[index] * (-1);
            }

            return -array[index];
        }

        [TestMethod]
        public void SLowFast()
        {
            int[] array = nd.ToArray<int>();
            int slow = 0;
            int fast = 0;

            while (true)
            {
                slow = array[slow];
                fast = array[array[fast]];
                if (slow == fast)
                    break;
            }

            fast = 0;
            while (true)
            {
                slow = array[slow];
                fast = array[fast];
                if (slow == fast)
                    break;
            }
        }

        private int Method4(int[] array)
        {
            int result = 0;
            for (int i = 1; i <= nd.size - 1; i++)
                result ^= i;
            for (int i = 0; i <= nd.size - 1; i++)
                result ^= array[i];

            return result;
        }
    }
}
