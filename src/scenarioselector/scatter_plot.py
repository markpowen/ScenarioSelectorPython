"""
Copyright 2021 Mark Philip Owen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from matplotlib import pyplot as plt
import numpy as np


class ScatterPlot:
    """A class which graphically shows the selection progress for problems with
    two variables."""

    def __init__(
            self,
            data,
            means=0,
            sums=0,
            annotate=None,
            xlims=None,
            ylims=None):
        data = np.asarray(data)
        self.trials, self.dimensions = data.shape
        if self.dimensions != 2:
            raise Exception("Scatter plot data must have two columns")
        self.data = data
        self.means = np.broadcast_to(means, self.dimensions)
        self.sums = np.broadcast_to(sums, self.dimensions)
        self.xlims = (np.min(data[:, 0]), np.max(
            data[:, 0])) if xlims is None else xlims
        self.ylims = (np.min(data[:, 1]), np.max(
            data[:, 1])) if ylims is None else ylims
        if annotate is None:
            self.annotate = True if self.trials <= 100 else False
        else:
            self.annotate = annotate

    def plot(self, lagrange_multiplier=(0, 0), fig_num=1):
        fig = plt.figure(fig_num, figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_xlim(self.xlims)
        ax.set_ylim(self.ylims)

        # Data and target
        selected = 1 - np.dot(self.data - self.means,
                              lagrange_multiplier) > -1.0e-10
        ax.scatter(self.data[:, 0], self.data[:, 1],
                   s=selected * 18 + 2, marker='o')
        ax.scatter(self.means[0], self.means[1], s=20, marker='x')
        ax.scatter(self.sums[0], self.sums[1], s=20, marker='x')
        for i in range(self.trials * self.annotate):
            ax.annotate(i, (self.data[i, 0], self.data[i, 1]))

        # Dividing line
        with np.errstate(divide='raise', invalid='raise'):
            try:
                xdata = np.array(self.xlims)
                ydata = (1 - lagrange_multiplier[0] * (xdata -
                                                       self.means[0]))
                ydata = ydata * lagrange_multiplier[1] + self.means[1]
                ax.plot(xdata, ydata, lw=2, color='blue')
                ydata = np.array(self.ylims)
                xdata = (1 - lagrange_multiplier[1] * (ydata -
                                                       self.means[1]))
                xdata = xdata / lagrange_multiplier[0] + self.means[0]
                ax.plot(xdata, ydata, lw=2, color='blue')
            except FloatingPointError:
                pass
        plt.show()
