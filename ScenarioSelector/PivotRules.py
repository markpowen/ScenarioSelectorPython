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

from ScenarioSelector.Exceptions import OptimizationIncompleteError
import numpy as np

__all__ = ['Dantzig', 'MaxObjectiveImprovement',
           'InsertLagrangeMultiplierIntoBasisThenDantzig', 'InsertLagrangeMultiplierIntoBasisThenMaxObjectiveImprovement']

def InsertLagrangeMultiplierIntoBasis(selector, start=0):
    """Permanently insert Lagrange multiplier variables into the basis, ejecting 'independent' trials in the process."""
    i = start
    for dimension in np.arange(selector.dimensions)[selector.multipliers_mask]:
        pivotElement = selector.PivotElement([dimension])
        if pivotElement is None: continue
        yield i, pivotElement
        i += 1

def LagrangeMultiplierDantzig(selector, start=0):
    dimensions = np.arange(selector.dimensions)[selector.multipliers_mask]
    i = start
    while dimensions.size:
        pivotElement = selector.PivotElement([dimensions[np.argmin(selector.ReducedCosts[dimensions])]])
        if pivotElement is None: return
        yield i, pivotElement
        i += 1
        dimensions = np.arange(selector.dimensions)[selector.multipliers_mask]

def LagrangeMultiplierMaxObjectiveImprovement(selector, start=0):
    dimensions = np.arange(selector.dimensions)[selector.multipliers_mask]
    i = start
    while dimensions.size:
        pivotElement = selector.PivotElement(dimensions)
        if pivotElement is None: return
        yield i, pivotElement
        i += 1
        dimensions = np.arange(selector.dimensions)[selector.multipliers_mask]

def Dantzig(selector, start=0, max_pivots=None):
    '''Dantzig's rule generally results in the smallest number of pivots because insertion of the Lagrange multiplier into the basis is delayed.'''
    if max_pivots is None: max_pivots = 2 * selector.trials
    i = start - 1
    for i in range(start, max_pivots):
        pivotElement = selector.PivotElement([np.argmin(selector.ReducedCosts)])
        if pivotElement is None: return
        yield i, pivotElement
    raise OptimizationIncompleteError("Exceeded maximum of {} pivot operations".format(max_pivots))

def MaxObjectiveImprovement(selector, start=0, max_pivots=None):
    if max_pivots is None: max_pivots = 4 * selector.trials
    i = start - 1
    for i in range(start, max_pivots):
        pivotElement = selector.PivotElement(np.arange(selector.dimensions))
        if pivotElement is None: return
        yield i, pivotElement
    raise OptimizationIncompleteError("Exceeded maximum of {} pivot operations".format(max_pivots))

def InsertLagrangeMultiplierIntoBasisThenDantzig(selector, start=0, max_pivots=None):
    if max_pivots is None: max_pivots = 2 * selector.trials
    i = start - 1
    for i, element in LagrangeMultiplierDantzig(selector, start): yield i, element
    yield from Dantzig(selector, i + 1, max_pivots)

def InsertLagrangeMultiplierIntoBasisThenMaxObjectiveImprovement(selector, start=0, max_pivots=None):
    if max_pivots is None: max_pivots = 2 * selector.trials
    i = start - 1
    for i, element in LagrangeMultiplierMaxObjectiveImprovement(selector, start): yield i, element
    yield from MaxObjectiveImprovement(selector, i + 1, max_pivots)
