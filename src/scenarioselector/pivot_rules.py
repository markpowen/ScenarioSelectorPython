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

from scenarioselector.pivot_dimensions import DantzigTwoPhase
from scenarioselector.pivot_trials import barrodale_roberts

__all__ = ['PivotRule', 'PivotRuleSlowed']


class PivotRule:
    """Returns the optimal pivot element from the supplied dimensions. The
    variable entering the basis corresponds to the integer
    self.independent_trials[pivot_dimension]. The variable leaving the basis
    corresponds to the integer pivot_trial."""

    def __init__(
            self,
            dimensions_generator=DantzigTwoPhase,
            pivot_trials=barrodale_roberts):
        self.dimensions_generator = dimensions_generator
        self.pivot_trials = pivot_trials

    def __call__(self, selector):
        dimensions_generator = self.dimensions_generator()
        for i, dimensions in dimensions_generator(selector):
            yield (i, *self.pivot_trials(selector, dimensions))


class PivotRuleSlowed(PivotRule):

    def __call__(self, selector):
        i = 0
        for _, (dimension, final_trial), flip_trials in super().__call__(
                selector):
            for trial in flip_trials:
                i += 1
                yield i, (dimension, trial), ()
            i += 1
            yield i, (dimension, final_trial), ()
