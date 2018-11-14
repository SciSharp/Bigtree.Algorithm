using MatplotlibCS;
using MatplotlibCS.PlotItems;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Bigtree.Algorithm.UnitTest.LinearRegression
{
    [TestClass]
    public class ModelTest
    {
        [TestMethod]
        public void Regression()
        {
            var np = new NumPy<double>();

            var data_x = np.linspace(1, 10, 100);
            var data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1);
            data_x /= np.max(data_x)[0];

            Plot(data_x, data_y);
        }

        private void Plot(NDArray<double> x, NDArray<double> y)
        {
            var matplotlibCs = new MatplotlibCS.MatplotlibCS("python", @"D:\Projects\MatplotlibCS\MatplotlibCS\Python\matplotlib_cs.py");

            var items = new List<PlotItem>();
            for(int i = 0; i < x.Size; i++)
            {
                items.Add(new Point2D($"P{i}", x[i], y[i])
                {
                    MarkerFaceColor = Color.Black,
                    MarkerSize = 3
                });
            }

            var figure = new Figure(1, 1)
            {
                FileName = "regression.png",
                OnlySaveImage = true,
                DPI = 150,
                Subplots =
                {
                    new Axes(1, "X axis", "Y axis")
                    {
                        Title = "Regression Test",
                        Grid = new Grid()
                        {
                            XLim = new double[] {0, 1.2},
                            YLim = new double[] {-2, 12}
                        },
                        PlotItems = items
                    }
                }
            };

            var t = matplotlibCs.BuildFigure(figure);
            t.Wait();
        }
    }
}
