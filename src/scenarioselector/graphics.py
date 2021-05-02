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
            annotate=None,
            lagrange_multiplier=None,
            fig_num=1,
            xlim=None,
            ylim=None):
        self.data = np.asarray(data)
        self.trials, self.dimensions = self.data.shape
        if self.dimensions != 2:
            raise Exception("Scatter plot data must have two columns")
        self.means = np.broadcast_to(means, self.dimensions)
        if xlim is None:
            self.xlim = (np.min(self.data[:, 0]), np.max(self.data[:, 0]))
        else:
            self.xlim = xlim 
        if ylim is None:
            self.ylim = (np.min(self.data[:, 1]), np.max(self.data[:, 1]))
        else:
            self.ylim = ylim
        if annotate is None:
            self.annotate = True if self.trials <= 100 else False
        else:
            self.annotate = annotate
        self.fig = plt.figure(fig_num, figsize=(7, 7))
        ax = self.fig.add_subplot(111)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        scatter_plot = ax.scatter(self.data[:, 0], self.data[:, 1], s=20,
                                  marker='o')
        target = ax.scatter(self.means[0], self.means[1], s=40, marker='x')
        line1, = ax.plot([], [], lw=2, color='blue')
        line2, = ax.plot([], [], lw=2, color='blue')
        self.components = [scatter_plot, target, line1, line2]
        self.update(lagrange_multiplier)
        plt.show()

    def update(self, multiplier):
        if multiplier is None:
            multiplier = (0, 0)
        scatter_plot = self.components[0]
        selected = 1.0 - np.dot(self.data - self.means, multiplier) > - 1.0e-10
        scatter_plot.set_sizes(selected * 18 + 2)
        line1 = self.components[2]
        line2 = self.components[3]
        with np.errstate(divide='raise', invalid='raise'):
            try:
                xdata = np.array(self.xlim)
                ydata = (1 - multiplier[0] * (xdata - self.means[0]))
                ydata = ydata / multiplier[1] + self.means[1]
            except FloatingPointError:
                xdata = []
                ydata = []
            line1.set_data(xdata, ydata)
            try:
                ydata = np.array(self.ylim)
                xdata = (1 - multiplier[1] * (ydata - self.means[1]))
                xdata = xdata / multiplier[0] + self.means[0]
            except FloatingPointError:
                xdata = []
                ydata = []
            line2.set_data(xdata, ydata)

    def animate(self, i, multipliers):
        self.update(multipliers[i])
        return self.components


class FanChart:
    """A class which graphically shows the selection progress for problems with
    multiple variables."""

    def __init__(
            self,
            num_trials,
            percentiles,
            means,
            levels,
            target_dimensions=[],
            target_mean=[],
            fig_num=1,
            xlim=None,
            ylim=None):
        self.trials = num_trials
        self.num_steps = np.shape(percentiles)[1]

        xlim = (0, self.num_steps) if xlim is None else xlim
        if ylim is None:
            min_y, max_y = np.min(percentiles), np.max(percentiles)
            ylim = (min_y, max_y)
        else:
            min_y, max_y = ylim

        self.fig = plt.figure(fig_num, figsize=(7, 4))

        self.ax = plt.axes(xlim=xlim, ylim=ylim)

        aspect_ratio = 9/16
        self.ax.set_aspect(aspect_ratio * self.num_steps / max_y)

        self.x = np.arange(0, self.num_steps)

        area1 = self.ax.fill_between(self.x,
                                     percentiles[5, :], percentiles[0, :],
                                     color='blue', alpha=5/50)
        area2 = self.ax.fill_between(self.x,
                                     percentiles[4, :], percentiles[1, :],
                                     color='blue', alpha=10/50)
        area3 = self.ax.fill_between(self.x,
                                     percentiles[3, :], percentiles[2, :],
                                     color='blue', alpha=25/50)

        mean_line, = self.ax.plot(np.arange(0, self.num_steps), means,
                                  color='blue')
        original_mean_line, = self.ax.plot(np.arange(0, self.num_steps), means,
                                           color='black')
        target_mean_line, = self.ax.plot(target_dimensions, target_mean,
                                         color='red')
        counter = self.ax.text(48, 0.075,
                               str(self.trials) + ' scenarios',
                               ha='right', va='top')

        self.components = [counter, mean_line, area1, area2, area3,
                           original_mean_line, target_mean_line]

        plt.show()

    def update(self, num_selected, percentiles, means):
        self.ax.collections.clear()
        area1 = self.ax.fill_between(self.x, percentiles[5, :],
                                     percentiles[0, :],
                                     color='blue', alpha=5/50)
        area2 = self.ax.fill_between(self.x, percentiles[4, :],
                                     percentiles[1, :],
                                     color='blue', alpha=10/50)
        area3 = self.ax.fill_between(self.x, percentiles[3, :],
                                     percentiles[2, :],
                                     color='blue', alpha=25/50)
        self.components[2] = area1
        self.components[3] = area2
        self.components[4] = area3

        mean_line = self.components[1]
        mean_line.set_data(self.x, means)
        counter = self.components[0]
        counter.set_text(str(num_selected) + ' scenarios')

    def animate(self, i, num_selected_list, percentiles_list, mean_list):
        self.update(num_selected_list[i], percentiles_list[i],  mean_list[i])
        return self.components
