The scenario selection algorithm selects a maximal subset of scenarios from a scenario set<sup>[1](#footnotes)</sup>, so that the selected scenarios have specified means (or sums).

## Package Installation

The Python scenarioselector package is published on the [Python Package Index](https://pypi.org/project/scenarioselector/), and is hosted on [GitHub](https://github.com/markpowen/ScenarioSelectorPython). The package can be installed with the [Package Installer for Python](https://pypi.org/project/pip/).

```
pip install scenarioselector
```

## Demonstrations

The easiest way to understand the scenario selection algorithm is to read through, download<sup>[2](#footnotes)</sup> and run two accompanying Jupyter Notebook demos presented with the [Jupyter NBViewer](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/tree/master/).

* [Demo 1: Selecting Scenarios from a Monte Carlo Simulation](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/blob/master/demo1.ipynb);
* [Demo 2: Selecting Scenarios of Bivariate Data](https://nbviewer.jupyter.org/github/markpowen/ScenarioSelectorNotebooks/blob/master/demo2.ipynb).

## Basic Usage

The following three steps outline basic usage of the ScenarioSelector class, which constructs instances of scenario selection problems and applies the selection algorithm.

### 1. Instantiate ScenarioSelector

Construct an object which defines the scenario selection problem you want to solve.

``` python
from scenarioselector import ScenarioSelector

selector = ScenarioSelector(data, weights=1, means=0, sums=0)
```

| Variable  | Allowable Types                                  | Shape      | Default Value                  | Description                                              |
|:----------|:-------------------------------------------------|:-----------|:-------------------------------|:---------------------------------------------------------|
| `data`    | List of lists, NumPy array, or pandas dataframe. | (_N_, _D_) | Required parameter.            | Scenario set with _N_ scenarios and _D_ variables.       |
| `weights` | Scalar, list or NumPy array.                     | (_N_,)     | Unit weight for each scenario. | Strictly positive weights for each of the _N_ scenarios. |
| `means`   | Scalar, list or NumPy array.                     | (_D_,)     | Zero mean for each variable.   | Target means for the _D_ variables.                      |
| `sums`    | Scalar, list or NumPy array.                     | (_D_,)     | Zero sum for each variable.    | Target sums for the _D_ variables.                       |

Note: Non-zero target values may be specified for either `means` or `sums`, but not both.

### 2. Run the Scenario Selection Algorithm

Call the ScenarioSelector's optimize method to run the scenario selection algorithm.

``` python
selector.optimize(callback=None, pivot_rule=None)
```

Note: Calling `selector.optimize()` without parameters runs the algorithm with default parameters.

### 3. View Results

Results of the optimization can be inspected as follows<sup>[3](#footnotes)</sup>.

* `selector.selected` is a Numpy array of Booleans which indicates which scenarios have been selected. If the input variable `data` is a NumPy array, then you can use NumPy's Boolean indexing functionality to obtain the selected scenario set as `selected_data = data[selector.selected]`, and the associated weights as `selected_weights = selector.weights[selector.selected]`.

	* If you have specified target `means`, the weighted means of the reduced scenario set will be close to your specified target. You can verify this by calculating `numpy.average(selected_data, weights=selected_weights, axis=0)`. If the original scenario set is equally weighted then you do not need to specify the selected weights.
	* If you have specified target `sums`, the weighted sums of the reduced scenario set will be close to your specified target. You can verify this by calculating `numpy.dot(selected_weights, selected_data)`. If each scenario has unit weight then you can get the same result by calculating `numpy.sum(selected_data, axis=0)`.

* `selector.reduced_weights` is a NumPy array of reduced weights associated with each scenario. You can verify the algorithm has hit the `sums` target precisely by calculating `numpy.dot(selector.reduced_weights, data)`.

* `selector.probabilities` is an NumPy array of probabilities associated with each scenario. You can verify the algorithm has hit the `means` target precisely by calculating `numpy.dot(selector.probabilities, data)`.

#### Example of Basic Usage

The following is an example of basic usage with _N_ = 5 and _D_ = 2.

Consider a finite discrete probability space, (_&Omega;_, P), where _&Omega;_ := {&omega;<sub>1</sub>, &omega;<sub>2</sub>, &omega;<sub>3</sub>, &omega;<sub>4</sub>, &omega;<sub>5</sub>}
and the probabilities of each outcome are _p_<sub>1</sub> = P(&omega;<sub>1</sub>) = 0.15, _p_<sub>2</sub> = P(&omega;<sub>2</sub>) = 0.25, _p_<sub>3</sub> = P(&omega;<sub>3</sub>) = 0.2, _p_<sub>4</sub> = P(&omega;<sub>4</sub>) = 0.25 and _p_<sub>5</sub> = P(&omega;<sub>5</sub>) = 0.15. 

Consider an R<sup>2</sup>-valued random variable _X_ with five realizations _X_(&omega;<sub>1</sub>) = (0.8, -3.2), _X_(&omega;<sub>2</sub>) = (3.0, 2.9), _X_(&omega;<sub>3</sub>) = (3.0, 2.5), _X_(&omega;<sub>4</sub>) = (-0.8, 1.0) and _X_(&omega;<sub>5</sub>) = (0.8, -2.0).

Suppose we want to select a maximal subset of the five scenarios, so that the weighted sum of the outcomes _X_(&omega;<sub>_n_</sub>) selected scenarios is equal to (1.1, 1.0). More precisely, we want to find reduced weights 0 &le; _q_<sub>_n_</sub> &le; _p_<sub>_n_</sub> which maximize &Sigma;<sub>_n_</sub> _q_<sub>_n_</sub>, subject to the constraint &Sigma;<sub>_n_</sub> _q_<sub>_n_</sub> _X_(&omega;<sub>_n_</sub>) = (1.1, 1.0).

We define an array of shape (5, 2) which holds the scenario set data.

```python
from scenarioselector import ScenarioSelector
import numpy as np

data    = np.array([[0.8, -3.2], [3.0, 2.9], [3.0, 2.5], [-0.8, -1.0], [0.8, -2.0]])
weights = [0.15, 0.25, 0.2, 0.25, 0.15]
sums    = [1.1, 1.0]

selector = ScenarioSelector(data, weights=weights, sums=sums)

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

Note: Python uses zero-based array indices so, for example, `data[1]` evaluates to `[3.0, 2.9]`.

## Advanced Usage

### ScenarioSelector Properties

A ScenarioSelector object has the following properties, which can be queried at any stage of the optimization.

| Property              | Type             | Shape  | Description                                                   |
|----------------------:|:------------------------|:-------|:--------------------------------------------------------------|
| `selected`            | NumPy array      | (_N_,) | Booleans, indicating which scenarios are selected.            |
| `reduced_weights`     | NumPy array      | (_N_,) | Reduced weights associated with each scenario.                |
| `probabilities`       | NumPy array      | (_N_,) | Probabilities associated with each scenario.                  |
| `lagrange_multiplier` | Numpy array      | (_D_,) | Lagrange multiplier for the dual problem.                     |
| `tableau`             | pandas dataframe |        | Condensed tableau for the simplex algorithm.                  |
| `pivot_count`         | int              |        | Number of pivots operations used to get to the current state. |

### Callback Function

The scenario selector's optimize method can be parameterized with a bespoke callback function. For example,

``` python
tableaus = []

def callback(selector, i, element):
	print("Iteration {} pivots on element {}.".format(i, element))
	tableaus.append(selector.tableau)
```

To keep track of the optimization progress, call the ScenarioSelector's optimize method with the callback function as a parameter.

``` python
selector.optimize(callback=callback)
```

### Pivot Rule

A pivot rule determines which variable and scenario(s) to use for pivot and flip operations in the modified simplex algorithm.

``` python
from scenarioselector.pivot_rule import PivotRule, PivotRuleSlowed
from scenarioselector.pivot_variable import (Dantzig, DantzigTwoPhase,
                                             MaxObjectiveImprovement, MaxObjectiveImprovementTwoPhase)
from scenarioselector.pivot_scenarios import pivot_scenarios, barrodale_roberts

pivot_rule = PivotRule(pivot_variable=DantzigTwoPhase, pivot_scenarios=barrodale_roberts)
selector.optimize(pivot_rule=pivot_rule)
```

The choices of pivot variable and pivot scenario(s) are discussed in the next two subsections.

Note: The derived pivot rule `PivotRuleSlowed` is designed specifically for use with the Barrodale Roberts improvement. This rule slows down the effect of passing through each vertex in succession, and is included only for demonstration purposes.

#### Pivot Variable

A `pivot_variable` rule determines which variable to use for the next pivot operation. Pre-defined `pivot_variable` rules can be summarised as follows.

| Rule                              | Description                                                                                                                       |
|----------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------|
| `Dantzig`                         | Choose the variable whose corresponding entry in the basement row of the condensed tableau has the largest magnitude.             |
| `DantzigTwoPhase`                 | Similar to `Dantzig`, however the first _D_ operations move all the Lagrange multiplier variables into the basis.                 |
| `MaxObjectiveImprovement`         | Choose the variable such that a classical pivot operation will lead to the largest improvement in the objective value.            |
| `MaxObjectiveImprovementTwoPhase` | Similar to `MaxObjectiveImprovement`, however the first _D_ operations move all the Lagrange multiplier variables into the basis. |

#### Pivot Scenarios

A `pivot_scenarios` rule determines which scenario(s) to use for the next pivot and associated flip operations. The Barrodale Roberts improvement allows the modified simplex algorithm to pass through multiple vertices at once, allowing the algorithm to flip an array of selection states in a single operation.

## Footnotes

1. A scenario set is a set of (possibly weighted) observations of multi-variate data.
1. The example notebooks are located in a separate project which is also hosted on [GitHub](https://github.com/markpowen/ScenarioSelectorNotebooks).
1. This section assumes you have imported NumPy with the statement `import numpy`.
