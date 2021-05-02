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

from scenarioselector.pivot_rule import PivotRule
import pandas as pd
import numpy as np

__all__ = ['ScenarioSelector']


class ScenarioSelector:
    """A class which selects as many scenarios as possible (more precisely, the
    maximum possible weight) from a data set, subject to constraints on
    means or sums for each variable (relative to the reduced maximal weights).

    Rows of the data set represent scenarios or observations.
    Columns of the data set represent the observed variables.
    If the data is not equally weighted, a set of weights can be supplied.

    Initialise an instance of the class with data, and call the optimize
    method. Post-optimization scenario weights are available as
    the property reduced_weights. See also the related property probabilities
    and the boolean array selected."""

    def __init__(self, data, weights=1, means=0, sums=0):
        """Sets up the initial simplex tableau for the linear program.

        Keyword arguments:
        data -- a dataframe or matrix whose columns represent variables, and
                whose rows represent 'observations' or 'weighted scenarios';
        weights -- weights for each observation/row of the data.
        means -- target means for each variable/column of the data;
        sums  -- target sums  for each variable/column of the data;
        """
        try:
            data = np.asarray(data)
            self.trials, self.dimensions = data.shape
        except ValueError:
            raise ValueError("data must be a dataframe or matrix.")
        try:
            self.weights = np.broadcast_to(weights, self.trials)
        except ValueError:
            raise ValueError("Length of weights must match the number of "
                             "observations (row dimension of data).")
        if means and sums:
            raise ValueError("You may specify either means or sums, "
                             "but not both.")
        try:
            means = np.broadcast_to(means, self.dimensions)
        except ValueError:
            raise ValueError("Length of means must match the number of "
                             "variables (column dimension of data).")
        try:
            sums = np.broadcast_to(sums, self.dimensions)
        except ValueError:
            raise ValueError("Length of sums must match the number of "
                             "variables (column dimension of data).")
        self.pivot_count = 0
        self.independent_trials = np.arange(-self.dimensions, 0)
        self.trials_mask = np.zeros(self.dimensions, dtype=bool)
        self.multipliers_mask = np.ones(self.dimensions, dtype=bool)
        # All scenarios are selected initially.
        self.selected = np.ones(self.trials, dtype=bool)
        adjusted_data = data - means
        self.tableau_array = np.block(
            [
                [-np.identity(self.dimensions),
                 np.zeros((self.dimensions, 1))],
                [adjusted_data, np.ones((self.trials, 1))],
                [self.weights @ adjusted_data - sums, self.weights.sum()]])

# %% Properties

    @property
    def tableau(self):
        """A labelled dataframe of the simplex tableau."""
        index = pd.MultiIndex.from_tuples(
            [('Lagrange Multiplier', i) for i in range(self.dimensions)] +
            [('Trial', i) for i in range(self.trials)] +
            [('Costs', '')], names=('Variable', 'Index'))
        columns = (pd.MultiIndex.from_tuples(
            [('Lagrange Multiplier', self.dimensions + i)
             if i < 0 else ('Trial', i) for i in self.independent_trials] +
            [('Feasible Solution', '')], names=('Variable', 'Index')))
        return pd.DataFrame(
            np.copy(self.tableau_array), index=index, columns=columns)

    @property
    def costs(self): return self.tableau_array[-1, :-1]

    @property
    def reduced_costs(self):
        reduced_costs = np.copy(self.costs)
        reduced_costs[self.trials_mask] += (self.selected[
            self.independent_trials[self.trials_mask]])
        return np.minimum(
            reduced_costs, self.trials_mask.astype(float) - reduced_costs)

    @property
    def adjusted_data(self):
        """Recovers the original adjusted data."""
        return -self.tableau_array[self.dimensions:-1, :-1] @ np.linalg.inv(
            self.tableau_array[:self.dimensions, :self.dimensions])

    @property
    def objective_value(self):
        """The dual problem objective value which is optimized by the linear
        program. This value coincides with a relaxed primal problem objective:
        np.sum(self.reduced_weights). On completion of the optimisation,
        0 <= self.reduced_weights <= self.weights."""
        return self.tableau_array[-1, -1]

    @property
    def lagrange_multiplier(self):
        """Lagrange multiplier is an array of length self.dimensions, which
        classifies the initial input data as
        Selected, Dependent:
            1 - adjusted_data[trial, :] @ lagrange_multiplier >  0;
        Deselected, Dependent:
            1 - adjusted_data[trial, :] @ lagrange_multiplier <  0;
        Independent:
            1 - adjusted_data[trial, :] @ lagrange_multiplier ~= 0;"""
        return np.copy(self.tableau_array[:self.dimensions, -1])

    @property
    def feasible_solution(self):
        """feasible_solution =
        1 - self.adjusted_data @ self.lagrange_multiplier"""
        return self.tableau_array[self.dimensions:-1, -1]

    @property
    def selected_trial_numbers(self):
        """Selected trial numbers."""
        return np.arange(self.trials)[self.selected]

    @property
    def reduced_weights(self):
        """Returns an array of length self.trials such that
        reduced_weights @ adjusted_data[:, dimension] = 0.
        Selected,   Dependent:          reduced_weights = weights;
        Deselected, Dependent:          reduced_weights = 0.0;
        Independent, post-opt:     0 <= reduced_weights <= weights.
        Following a complete optimisation,
        self.objective_value =~ np.sum(self.reduced_weights[:self.trials]).
        and
        np.dot(selector.reduced_weights,
        np.concatenate((adjusted_data, -np.identity(dimensions)))) = 0."""
        reduced_weights = np.concatenate(
            (self.selected.astype(float) * self.weights, np.zeros(
                self.dimensions, dtype=float)))
        reduced_weights[self.independent_trials] += self.costs
        return reduced_weights[:self.trials]

    @property
    def probabilities(self):
        """Candidate weights for each trial. Once the Lagrange multiplier
        variables have been inserted into the basis, we have
        probabilities @ data = target_means."""
        return self.reduced_weights / self.objective_value

# %% Simplex Tableau Operations

    def set_state_lagrange_multiplier(self, lagrange_multiplier):
        """Fast-forward the optimization of a ScenarioSelector instance using a
        previously computed Lagrange multiplier."""
        if len(lagrange_multiplier) != self.dimensions:
            raise Exception('lagrange_multiplier must be an array of length ' +
                            str(self.dimensions))
        independent_trials = np.argsort(np.abs(1.0 - np.dot(
            self.adjusted_data, lagrange_multiplier)))[:self.dimensions]
        return self.set_state_independent_trials(independent_trials)

    def set_state_independent_trials(self, independent_trials):
        """Fast-forward the optimization of a ScenarioSelector instance using a
        previously computed list of trials to eject from the basis (assigning
        independent weights)."""
        independent_trials = np.array(independent_trials)
        if len(independent_trials) != self.dimensions or not (
            np.all(independent_trials < self.trials) and np.all(
                independent_trials >= 0)):
            raise Exception(
                'independent_trials must be a list of {} indices, '
                'numbered between 0 and {}.'.format(
                    self.dimensions, self.trials - 1))
        for pivot_dimension in range(self.dimensions):
            self.Pivot((pivot_dimension, independent_trials[pivot_dimension]))
        return self.flip_trials(np.arange(self.trials)[np.logical_xor(
            self.feasible_solution > 0,
            self.selected)]).fix_feasible_solution()

    def optimize(self, callback=None, pivot_rule=None):
        """Solves the linear program using the simplex method.

        Keyword arguments:
        callback -- externally monitor optimisation progress;
        pivot_generator -- pivot rule implemented as a python generator.
        """
        if pivot_rule is None:
            pivot_rule = PivotRule()
        if callback is None:
            def callback(selector, i, element):
                return None
        with np.errstate(divide='ignore', invalid='ignore'):
            callback(self, 0, (None, None))
            for self.pivot_count, element, trials in pivot_rule(self):
                self.pivot(element).flip_trials(trials).fix_feasible_solution()
                callback(self, self.pivot_count, element)
            return self  # Optimisation complete

    def pivot(self, element):
        """Pivot the simplex tableau.
        Vvariable about to enter the basis: self.independent_trials[dimension].
        Variable about to leave the basis: trial."""
        dimension, trial = element
        self.independent_trials[dimension] = trial
        self.trials_mask[dimension] = True
        self.multipliers_mask[dimension] = False
        row, columns = self.dimensions + trial, np.delete(
            np.arange(self.dimensions + 1), dimension)
        pivot_column_data = self.tableau_array[:, dimension]
        pivot_column_data *= -1.0 / self.tableau_array[row, dimension]
        self.tableau_array[:, columns] += np.outer(
            pivot_column_data, self.tableau_array[row, columns])
        return self

    def flip_trials(self, trials=()):
        """Flip (deselect/reselect) independent trials. This method is designed
        to have no impact on the ReducedWeights property."""
        if len(trials):
            self.selected[trials] = np.logical_not(self.selected[trials])
            weights = self.weights[trials]
            self.tableau_array[-1, :] += np.where(
                self.selected[trials], weights, -weights) @ self.tableau_array[
                self.dimensions + trials, :]
        independent_trials = self.independent_trials[self.trials_mask]
        selected_independent_trials = (
            self.selected[independent_trials] + self.costs[self.trials_mask] /
            self.weights[independent_trials] > 0.5)
        self.tableau_array[-1, :] += (
            (
                np.subtract(selected_independent_trials,
                            self.selected[independent_trials], dtype=int) *
                self.weights[independent_trials])
            @ self.tableau_array[self.dimensions + independent_trials, :])
        self.selected[independent_trials] = selected_independent_trials
        return self

    def fix_feasible_solution(self):
        feasible_solution = self.feasible_solution
        feasible_solution[np.logical_and(
            self.selected, feasible_solution <= 0)] = 0.0
        feasible_solution[np.logical_and(np.logical_not(
            self.selected), feasible_solution >= 0)] = -0.0
        return self
