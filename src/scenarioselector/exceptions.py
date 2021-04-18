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

__all__ = ['DegenerateLinearProgramError', 'OptimizationIncompleteError']


class ScenarioSelectorError(Exception):
    """Base class for exceptions in this module."""
    pass


class DegenerateLinearProgramError(ScenarioSelectorError):
    """Target does not lie within the convex hull of the observations."""

    def __init__(self):
        super().__init__("Target does not lie within the convex hull " +
                         "of the observations.")


class OptimizationIncompleteError(ScenarioSelectorError):
    """Optimization incomplete."""

    def __init__(self, max_pivots):
        super().__init__("Exceeded maximum of " + str(max_pivots) +
                         " pivot operations")
