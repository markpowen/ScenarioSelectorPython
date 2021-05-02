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

from scenarioselector.exceptions import (OptimizationIncompleteError, 
                                         DegenerateLinearProgramError)
import numpy as np

__all__ = ['Dantzig', 'DantzigTwoPhase',
           'MaxObjectiveImprovement', 'MaxObjectiveImprovementTwoPhase']


class DimensionsGenerator:

    def __init__(self, start=1, max_pivots=None, tolerance=1.0e-10):
        self.start = start
        self.max_pivots = max_pivots
        self.tolerance = tolerance

    def check_for_degeneracy(self, selector):
        if not(selector.objective_value > self.tolerance):
            raise DegenerateLinearProgramError()

    def optimization_incomplete(self):
        raise OptimizationIncompleteError(self.max_pivots)


class Dantzig(DimensionsGenerator):

    def __call__(self, selector):
        if self.max_pivots is None:
            self.max_pivots = 2 * selector.trials
        for i in range(self.start, self.max_pivots):
            self.check_for_degeneracy(selector)
            dimension = np.argmin(selector.reduced_costs)
            if selector.reduced_costs[dimension] < -self.tolerance:
                yield i, (dimension,)
            else:
                return
        self.optimization_incomplete()


class DantzigTwoPhase(Dantzig):

    def __call__(self, selector):
        num_multipliers = np.sum(selector.multipliers_mask)
        dimensions = np.arange(selector.dimensions)
        i = self.start - 1
        for i in self.start + np.arange(num_multipliers):
            self.check_for_degeneracy(selector)
            dimension = dimensions[selector.multipliers_mask][np.argmin(
                selector.reduced_costs[selector.multipliers_mask])]
            if selector.reduced_costs[dimension] < -self.tolerance:
                yield i, (dimension,)
            else:
                break
        self.start = i + 1
        yield from super().__call__(selector)


class MaxObjectiveImprovement(DimensionsGenerator):

    def __call__(self, selector):
        if self.max_pivots is None:
            self.max_pivots = 4 * selector.trials
        dimensions = np.arange(selector.dimensions)
        for i in range(self.start, self.max_pivots):
            self.check_for_degeneracy(selector)
            dimensions_subset = dimensions[
                selector.reduced_costs < -self.tolerance]
            if len(dimensions_subset):
                yield i, dimensions_subset
            else:
                return
        self.optimization_incomplete()


class MaxObjectiveImprovementTwoPhase(MaxObjectiveImprovement):

    def __call__(self, selector):

        num_multipliers = np.sum(selector.multipliers_mask)
        dimensions = np.arange(selector.dimensions)
        i = self.start - 1
        for i in self.start + np.arange(num_multipliers):
            self.check_for_degeneracy(selector)
            dimensions_subset = dimensions[np.logical_and(
                selector.multipliers_mask, selector.reduced_costs <
                -self.tolerance)]
            if len(dimensions_subset):
                yield i, dimensions_subset
            else:
                break
        self.start = i + 1
        yield from super().__call__(selector)
