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

from ScenarioSelector.Exceptions import DegenerateLinearProgramError
from ScenarioSelector.PivotRules import InsertLagrangeMultiplierIntoBasisThenDantzig
import pandas as pd
import numpy as np; np.seterr(divide='ignore', invalid='ignore')

__all__ = ['ScenarioSelector']

class ScenarioSelector:
    """A class which selects the largest number of observations of variables from an input data set, such that target means are hit.
    To use this class, initialize data and target_means, and call the method self.Optimize().
    Post-optimization scenario weights are available as the property self.ScalingFactors.
    See also the related property self.Probabilities and the boolean array self.selected."""

    def __init__(self,
                 data, # A matrix in which columns represent variables and rows represent observations.
                 target_means=None): # Target means for each column of the input data.
        """Sets up the initial simplex tableau for the linear program."""
        data = np.asarray(data)
        self.trials, self.dimensions = data.shape
        target_means = np.zeros(self.dimensions) if target_means is None else np.asarray(target_means)
        if len(target_means) != self.dimensions: raise Exception("Length of target means does not match the columns of input data.")
        adjustedData = data - target_means
        self.independent_trials = np.arange(-self.dimensions, 0)
        self.trials_mask = np.zeros(self.dimensions, dtype=bool)
        self.multipliers_mask = np.ones(self.dimensions, dtype=bool)
        self.selected = np.ones(self.trials, dtype=bool) # Indicates whether scenarios are selected.
        self.tableau = np.block([[ -np.identity(self.dimensions), np.zeros((self.dimensions, 1)) ],
                                 [  adjustedData,                 np.ones((self.trials, 1))      ],
                                 [  adjustedData.sum(axis=0),     self.trials                    ]])

    ##############
    # PROPERTIES #
    ##############

    @property
    def Tableau(self):
        """A labelled dataframe of the simplex tableau."""
        index   = pd.MultiIndex.from_tuples([('Lagrange Multiplier', i) for i in range(self.dimensions)] + [('Trial', i) for i in range(self.trials)] + [('Costs', '')], names=('Variable', 'Index'))
        columns = pd.MultiIndex.from_tuples([('Lagrange Multiplier', self.dimensions + i) if i < 0 else ('Trial', i) for i in self.independent_trials] + [('Feasible Solution', '')], names=('Variable', 'Index'))
        return pd.DataFrame(np.copy(self.tableau), index=index, columns=columns)

    @property
    def Costs(self): return self.tableau[-1, :-1]

    @property
    def ReducedCosts(self):
        reducedCosts = np.copy(self.Costs)
        reducedCosts[self.trials_mask] += self.selected[self.independent_trials[self.trials_mask]]
        return np.minimum(reducedCosts, self.trials_mask.astype(float) - reducedCosts)

    @property
    def AdjustedData(self):
        """Recovers the original adjusted data."""
        return -self.tableau[self.dimensions:-1, :-1] @ np.linalg.inv(self.tableau[:self.dimensions, :self.dimensions])

    @property
    def ObjectiveValue(self):
        """The dual problem objective value which is optimized by the linear program.
        This value coincides with a relaxed primal problem objective: np.sum(self.ScalingFactors).
        On completion of the optimisation, 0 <= self.ScalingFactors <= 1."""
        return self.tableau[-1, -1]

    @property
    def LagrangeMultiplier(self):
        """Lagrange multiplier is an array of length self.dimensions, which classifies the initial input data as
        Selected,   Dependent: 1 - adjustedData[trial, :] @ LagrangeMultiplier >  0;
        Deselected, Dependent: 1 - adjustedData[trial, :] @ LagrangeMultiplier <  0;
        Independent:           1 - adjustedData[trial, :] @ LagrangeMultiplier ~= 0;"""
        return np.copy(self.tableau[:self.dimensions, -1])

    @property
    def FeasibleSolution(self):
        """FeasibleSolution = 1 - self.adjustedData @ self.LagrangeMultiplier"""
        return self.tableau[self.dimensions:-1, -1]

    @property
    def NumberSelected(self): return np.sum(self.selected)
    
    @property
    def ScalingFactors(self):
        """Scaling factors is an array of length self.trials such that
        ScalingFactors @ adjustedData[:, dimension] = 0.
        Selected,   Dependent:          ScalingFactor = 1.0;
        Deselected, Dependent:          ScalingFactor = 0.0;
        Independent, post-optimisation: ScalingFactor \in [0,1].
        Following a complete optimisation, self.ObjectiveValue =~ np.sum(self.ScalingFactors[:self.trials]).
        and
        np.dot(selector.ScalingFactors, np.concatenate((adjusted_data, -np.identity(dimensions)))) = 0."""
        scalingFactors = np.concatenate((self.selected.astype(float), np.zeros(self.dimensions, dtype=float)))
        scalingFactors[self.independent_trials] += self.Costs
        return scalingFactors

    @property
    def Probabilities(self):
        """Candidate probabilities for each trial.
        Once the Lagrange multiplier variables have been inserted into the basis, we have
        Probabilities @ data = target_means."""
        return self.ScalingFactors[:self.trials] / self.ObjectiveValue

    ##############################
    # SIMPLEX TABLEAU OPERATIONS #
    ##############################

    def SetStateLagrangeMultiplier(self, lagrange_multiplier):
        """Fast-forward the optimization of a ScenarioSelector instance using a previously computed Lagrange multiplier."""
        if len(lagrange_multiplier) != self.dimensions: raise Exception('lagrange_multiplier must be an array of length ' + str(self.dimensions))
        independent_trials = np.argsort(np.abs(1.0 - np.dot(self.AdjustedData, lagrange_multiplier)))[:self.dimensions]
        return self.SetStateIndependentTrials(independent_trials)

    def SetStateIndependentTrials(self, independent_trials):
        """Fast-forward the optimization of a ScenarioSelector instance using a previously computed list of trials to eject from the basis (assigning independent weights)."""
        independent_trials = np.array(independent_trials)
        if len(independent_trials) != self.dimensions or not (np.all(independent_trials < self.trials) and np.all(independent_trials >= 0)):
            raise Exception('independent_trials must be a list of {} indices, numbered between 0 and {}.'.format(self.dimensions, self.trials - 1))
        for pivotDimension in range(self.dimensions): self.Pivot((pivotDimension, independent_trials[pivotDimension]))
        self.selected = self.FeasibleSolution > 0
        self.tableau[-1, :] = np.sum(self.tableau[self.dimensions + np.arange(self.trials)[self.selected], :], axis=0)
        return self.FlipIndependentTrials().FixFeasibleSolution()

    def Optimize(self,
                 callback = None, # Externally monitor optimisation progress.
                 pivotRule = None, # Specify a pivot rule, determining which variables enter and leave the basis.
                 max_pivots = None): # Manually specify the maximum number of iterations, or wrap with a progress bar.
        """Solves the linear program using the simplex method."""
        if callback is None: callback = lambda scenarioSelector, i, element: None
        if pivotRule is None: pivotRule = InsertLagrangeMultiplierIntoBasisThenDantzig
        
        i = -1
        for i, element in pivotRule(self, 0, max_pivots):
            callback(self, i, element)
            self.Pivot(element).FlipIndependentTrials().FixFeasibleSolution()
        callback(self, i + 1, element)
        self.pivot_count = i + 1
        return self # Optimisation complete

    def Pivot(self, element):
        """Pivot the simplex tableau.
        The variable about to enter the basis: self.independent_trials[dimension].
        The variable about to leave the basis: trial."""
        dimension, trial = element
        self.independent_trials[dimension], self.trials_mask[dimension], self.multipliers_mask[dimension] = trial, True, False
        row, columns = self.dimensions + trial, np.delete(np.arange(self.dimensions + 1), dimension)
        pivotColumnData = self.tableau[:, dimension]
        pivotColumnData *= -1.0 / self.tableau[row, dimension]
        self.tableau[:, columns] += np.outer(pivotColumnData, self.tableau[row, columns])
        return self

    def FlipIndependentTrials(self):
        """Flip (deselect/reselect) independent trials.
        This method is designed to have no impact on the ScalingFactors property."""
        independent_trials = self.independent_trials[self.trials_mask]
        selected_independent_trials = self.Costs[self.trials_mask] + self.selected[independent_trials] > 0.5
        self.tableau[-1, :] += np.subtract(selected_independent_trials, self.selected[independent_trials], dtype=int) @ self.tableau[self.dimensions + independent_trials, :]
        self.selected[independent_trials] = selected_independent_trials
        return self

    def FixFeasibleSolution(self):
        feasible_solution = self.FeasibleSolution
        feasible_solution[np.logical_and(self.selected, feasible_solution <= 0)] = 0.0
        feasible_solution[np.logical_and(np.logical_not(self.selected), feasible_solution >= 0)] = -0.0
        return self

    def PivotElement(self, dimensions, tolerance=1.0e-10):
        """Returns the optimal pivot element from the supplied dimensions.
        The variable entering the basis corresponds to the integer self.independent_trials[pivotDimension].
        The variable leaving  the basis corresponds to the integer pivotTrial."""
        if not(self.ObjectiveValue > tolerance): raise DegenerateLinearProgramError("Target means do not lie within the convex hull of the rows of data.")
        dimensions = np.asarray(dimensions)[self.ReducedCosts[dimensions] < -tolerance]
        if len(dimensions) == 0: return None
        ratios = self.tableau[self.dimensions:-1, dimensions] / np.outer(self.FeasibleSolution, self.Costs[dimensions])
        ratios[self.independent_trials[self.trials_mask]] = np.nan
        pivotTrials = np.nanargmax(ratios, axis=0)
        dimension_index = np.argmin(ratios[pivotTrials, np.arange(len(dimensions))])
        pivotDimension, pivotTrial = dimensions[dimension_index], pivotTrials[dimension_index]
        return pivotDimension, pivotTrial
