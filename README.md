The ScenarioSelector algorithm selects a maximal subset of scenarios from a set of weighted observations of multi-variate data, so that the multivariate means (or sums) of the reduced subset of "selected" scenarios hit a specified target.

## Package Installation

The scenarioselector package is published on [PyPI](https://pypi.org/project/scenarioselector/), and hosted on [GitHub](https://github.com/markpowen/ScenarioSelectorPython).

```
pip install scenarioselector
```

### Dependencies

* `numpy` (version &GreaterEqual; 1.18.0)
* `pandas` (version &GreaterEqual; 1.0.0)

## Examples

The easiest way to understand the scenario selection algorithm is to read through and run two accompanying Jupyter notebook examples, which can be found on [jupyter.org](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/tree/master/) as follows.

1. [Selecting Scenarios of Bivariate Data](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/blob/master/Example1.ipynb);
1. [Selecting Scenarios from a Monte Carlo Simulation](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/blob/master/Example2.ipynb).

## Basic Usage

The following three steps outline basic usage of the ScenarioSelector class.

### 1. Instantiate ScenarioSelector Class

Construct an object which defines the scenario selection problem you wish to solve.

``` python
from scenarioselector import ScenarioSelector

selector = ScenarioSelector(data, weights=1, means=0, sums=0)
```

| Variable  | Description                                                                               | Type                                   | Shape |
|-----------|:-------------------------------------------------------------------------------------------------|----------------------------------------|---|
| `data`    | Scenario set data with _N_ scenarios and _D_ variables. | Array, numpy array or pandas dataframe. |  (_N_, _D_) |
| `weights` | Strictly positive weights for each of the _N_ scenarios. The default value gives unit weight to each scenario. | Scalar, array or numpy array.           | (_N_,) |
| `means`   | Target means for the _D_ variables.                                                         | Scalar, array or numpy array.           | (_D_,)|
| `sums`    | Target sums for the _D_ variables.                                                          | Scalar, array or numpy array.           | (_D_,) |

Note: Values may be specified for _at most one_ of the input variables `means` and `sums`.

### 2. Run the Scenario Selection Algorithm

Call the optimize method to run the scenario selection algorithm with default settings.

``` python
selector.optimize()
```

### 3. View Results

Results of the optimization can be inspected as follows.

* `selector.selected` is a Boolean array which indicates which scenarios have been selected. If `data` is a numpy array, then you can use numpy's Boolean indexing functionality to obtain the reduced scenario set as `reduced_data = data[selector.selected]`.

	* When you specify a target `sums` vector, the weighted sums of the reduced scenario set will be close to your specified target vector. You can verify this by calculating `np.dot(reduced_data, selector.weights[selector.selected])`.
	* When you specify a target `means` vector, the weighted means of the reduced scenario set will be close to your specified target vector. You can verify this by calculating `np.average(reduced_data, weights=selector.weights[selector.selected])`.

* `selector.reduced_weights` is an array of reduced weights associated with each scenario. You can verify the algorithm has hit a `sums` target precisely by calculating `np.dot(selector.reduced_weights, data)`.

* `selector.probabilities` is an array of probabilities associated with each scenario. You can verify the algorithm has hit the `means` target precisely by calculating `np.dot(selector.probabilities, data)`.

#### Example of Basic Usage

The following example has _N_ = 5 and _D_ = 2.

Consider a finite discrete probability space, (_&Omega;_, P), where _&Omega;_ := {&omega;<sub>1</sub>, &omega;<sub>2</sub>, &omega;<sub>3</sub>, &omega;<sub>4</sub>, &omega;<sub>5</sub>}
and the probabilities of each outcome are _p_<sub>1</sub> = P(&omega;<sub>1</sub>) = 0.15, _p_<sub>2</sub> = P(&omega;<sub>2</sub>) = 0.25, _p_<sub>3</sub> = P(&omega;<sub>3</sub>) = 0.2, _p_<sub>4</sub> = P(&omega;<sub>4</sub>) = 0.25, _p_<sub>5</sub> = P(&omega;<sub>5</sub>) = 0.15. 

Consider an R<sup>2</sup>-valued random variable _X_ with five realizations _X_(&omega;<sub>1</sub>) = (0.8, -3.2), _X_(&omega;<sub>2</sub>) = (3.0, 2.9), _X_(&omega;<sub>3</sub>) = (3.0, 2.5), _X_(&omega;<sub>4</sub>) = (-0.8, 1.0), _X_(&omega;<sub>5</sub>) = (0.8, -2.0).

Suppose we want to select a maximal subset of the five scenarios, so that the weighted sum of the outcomes _X_(&omega;<sub>_n_</sub>) selected scenarios is equal to (1.1, 1.0). More precisely, we want to find reduced weights 0 &le; _q_<sub>_n_</sub> &le; _p_<sub>_n_</sub> which maximize &Sigma;<sub>_n_</sub> _q_<sub>_n_</sub>, subject to the constraint &Sigma;<sub>_n_</sub> _q_<sub>_n_</sub> _X_(&omega;<sub>_n_</sub>) = (1.1, 1.0).

We define an array of shape (5, 2) which holds the scenario set data.

```python
from scenarioselector import ScenarioSelector
import numpy as np

data    = np.array([[0.8, -3.2], [3.0, 2.9], [3.0, 2.5], [-0.8, -1.0], [0.8, -2.0]])
weights = [0.15, 0.25, 0.2, 0.25, 0.15]
sums    = [1.1, 1.0]

selector = ScenarioSelector(data, weights=weights, sum=sums)

print()
print("Before optimization")
print("-------------------")
print(sum(selector.selected), "scenarios selected: ", selector.selected)
print("Exact sums:", np.dot(selector.reduced_weights, data))
print("Approx sums:", np.dot(selector.weights[selector.selected], data[selector.selected]))

selector.optimize()

print()
print("After optimization")
print("------------------")
print(sum(selector.selected), "scenarios selected: ", selector.selected)
print("Exact sums:", np.dot(selector.reduced_weights, data))
print("Approx sums:", np.dot(selector.weights[selector.selected], data[selector.selected]))
```

Note that python uses zero-based array indices so, for example, `data[1]` evaluates to `[3.0, 2.9]`.

## Advanced Usage

The scenario selector's optimize method can be parameterized with a callback function and/or a pivot rule.

``` python
selector.optimize(callback=None, pivot_rule=None)
```

### Callback Function

Define a callback function as follows.

``` python
def callback(selector, i, element):
	print("On iteration {}, we pivot on element {}, leading to the condensed tableau".format(i, element))
	print(selector.tableau)
```

To keep track of the optimization progress, call the scenario selector's optimize method with the callback function as a parameter.

``` python
selector.optimize(callback=callback)
```

### ScenarioSelector Properties

A ScenarioSelector object has the following properties, which can be queried at any stage of the optimization.

| Property              | Description                                                             |
|----------------------:|:------------------------------------------------------------------------|
| `probabilities`       | An array of _N_ probabilities associated with each scenario.            |
| `reduced_weights`     | An array of _N_ reduced weights associated with each scenario.          |
| `selected`            | A Boolean array of length _N_, indicating which scenarios are selected. |
| `tableau`             | Condensed tableau for the simplex algorithm.                            |
| `pivot_count`         | Number of pivots operations used to get to the current state.           |
| `lagrange_multiplier` | Lagrange multiplier for the dual problem.                               |

### Pivot Rules

A pivot rule determines which variable and scenario(s) to use for the next pivot and flip operations in the modified simplex algorithm. A pivot rule comprises a 'Pivot Variable' rule and a 'Pivot Scenarios' rule.

#### Pivot Variable

A `pivot_variable` rule determines which variable to use for the next pivot operation. A selection of pre-defined `pivot_variable` rules can be imported as follows.

``` python
from scenarioselector.pivot_variable import (Dantzig, DantzigTwoPhase,
                                             MaxObjectiveImprovement, MaxObjectiveImprovementTwoPhase)
```

Pre-defined `pivot_variable` rules can be summarised as follows.

| Rule                              | Description                                                                                                            |
|----------------------------------:|------------------------------------------------------------------------------------------------------------------------|
| `Dantzig`                         | Choose the variable whose corresponding entry in the basement row of the condensed tableau has the largest magnitude.  |
| `DantzigTwoPhase`                 | Similar to `Dantzig`, however the first _D_ operations move all the Lagrange multiplier variables into the basis.      |
| `MaxObjectiveImprovement`         | Choose the variable such that a classical pivot operation will lead to the largest improvement in the objective value. |
| `MaxObjectiveImprovementTwoPhase` | Similar to `MaxObjectiveImprovement`, however the first _D_ operations move all the Lagrange multiplier variables into the basis. |

#### Pivot Scenarios

A `pivot_scenarios` rule determines which scenario(s) to use for the next pivot and associated flip operations. A selection of pre-defined `pivot_scenario` rules can be imported as follows.

``` python
from scenarioselector.pivot_scenarios import pivot_scenarios, barrodale_roberts
```

The Barrodale Roberts improvement allows the algorithm to pass through multiple vertices at once, allowing the algorithm to flip an array of selection states in a single operation.

#### Pivot Rule

To construct a pivot rule, we combine together a pivot_variable and a pivot_scenarios rule as follows.

You may choose between two classes of pivot rule: `PivotRule` and `PivotRuleSlowed`. PivotRules are constructed as a combination of a `pivot_variable` rule and a `pivot_scenarios` rule.

``` python
from scenarioselector.pivot_rules import PivotRule, PivotRuleSlowed

pivot_rule = PivotRule(pivot_variable=DantzigTwoPhase, pivot_scenarios=barrodale_roberts)

selector.optimize(pivot_rule=pivot_rule)
```

The default `pivot_variable` rule is `DantzigTwoPhase`, and the default `pivot_scenarios` rule is `barrodale_roberts`.

NB: `PivotRuleSlowed` is designed specifically for use with the `barrodale_roberts` `pivot_scenario` rule, as it demonstrates the effect of passing through each vertex in succession.
