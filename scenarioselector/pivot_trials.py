"""
Copyright 2017 Mark Philip Owen

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

from scenarioselector.exceptions import DegenerateLinearProgramError

import numpy as np

__all__ = ['trials_chooser', 'trials_chooser_barrodale_roberts']


def trials_chooser(selector, dimensions):
    ratios = (selector.tableau_array[selector.dimensions:-1, dimensions] /
              np.outer(selector.feasible_solution, selector.costs[dimensions]))
    ratios[selector.independent_trials[selector.trials_mask]] = np.nan
    try:
        pivot_trials = np.nanargmax(ratios, axis=0)
    except ValueError:
        raise DegenerateLinearProgramError()
    dimensionIndex = np.argmin(ratios[pivot_trials,
                                      np.arange(len(dimensions))])
    return (dimensions[dimensionIndex], pivot_trials[dimensionIndex]), ()


def trials_chooser_barrodale_roberts(selector, dimensions):
    data = selector.tableau_array[selector.dimensions:-1, dimensions]
    costs = selector.costs[dimensions]
    weights = selector.weights
    ratios = data / np.outer(selector.feasible_solution, costs)
    ratios[selector.independent_trials[selector.trials_mask]] = np.nan
    sorted_trials = np.argsort(-ratios, axis=0)
    sorted_trial_indices = np.argmax(np.divide(np.take_along_axis(np.multiply(
        data, np.where(selector.selected, weights, -weights)[:, np.newaxis]),
        sorted_trials, axis=0).cumsum(axis=0), costs) >= 1, axis=0)
    pivot_trials = sorted_trials[sorted_trial_indices,
                                 np.arange(len(dimensions))]
    dimension_index = np.argmin(ratios[pivot_trials,
                                       np.arange(len(dimensions))])
    return (dimensions[dimension_index], pivot_trials[dimension_index]
            ), sorted_trials[
                :sorted_trial_indices[dimension_index], dimension_index]
