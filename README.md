# PyDTMC

PyDTMC is a full-featured, lightweight library for discrete-time Markov chains analysis. It provides classes and functions for creating, manipulating and simulating markovian stochastic processes.

## Requirements

PyDTMC supports only `Python 3` and the minimum required version is `3.6`. In addition, the environment must include the following libraries:

* [Matplotlib](https://matplotlib.org/)
* [NetworkX](https://networkx.github.io/)
* [Numpy](https://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

For a better user experience, it's recommended to install [Graphviz](https://www.graphviz.org/) and [PyDot](https://pypi.org/project/pydot/) before using the `plot_graph` function.
In order to build the project documentation, it's necessary to install [Sphinx](https://www.sphinx-doc.org/).
In order to perform unit tests, it's necessary to install [PyTest](https://pytest.org/).


## Installation & Upgrade

Via PyPI:

```sh
$ pip install PyDTMC
$ pip install --upgrade PyDTMC
```

Via GitHub:

```sh
$ pip install git+https://github.com/TommasoBelluzzo/PyDTMC.git@master#egg=PyDTMC
$ pip install --upgrade git+https://github.com/TommasoBelluzzo/PyDTMC.git@master#egg=PyDTMC
```

## Usage

The core element of the library is the `MarkovChain` class, which can be instantiated as follows:

```console
>>> p = [[0.2, 0.7, 0.0, 0.1], [0.0, 0.6, 0.3, 0.1], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0]]
>>> mc = MarkovChain(p, ['A', 'B', 'C', 'D'])
>>> print(mc)

DISCRETE-TIME MARKOV CHAIN
 SIZE:           4
 RANK:           4
 CLASSES:        2
  > RECURRENT:   1
  > TRANSIENT:   1
 ERGODIC:        NO
  > APERIODIC:   YES
  > IRREDUCIBLE: NO
 ABSORBING:      YES
 REGULAR:        NO
 REVERSIBLE:     NO
```

Below a few examples of `MarkovChain` instance properties and static computations:

```console
>>> print(mc.is_ergodic)
False

>>> print(mc.recurrent_states)
['C']

>>> print(mc.transient_states)
['A', 'B', 'D']

>>> print(mc.steady_states)
[array([0.0, 0.0, 1.0, 0.0])]

>>> print(mc.is_absorbing)
True

>>> print(mc.fundamental_matrix)
[[1.50943396 2.64150943 0.41509434]
 [0.18867925 2.83018868 0.30188679]
 [0.75471698 1.32075472 1.20754717]]
 
>>> print(mc.kemeny_constant)
5.547169811320755

>>> print(mc.mean_absorption_times())
[4.56603774 3.32075472 3.28301887]

>>> print(mc.absorption_probabilities())
[1.0 1.0 1.0]

>>> print(mc.entropy_rate)
0.0
```

Dynamic computations on `MarkovChain` instances can be performed through their parametrized methods:

```console
>>> print(mc.expected_rewards(10, [2, -3, 8, -7]))
[-2.76071635, -12.01665113, 23.23460025, -8.45723276]

>>> print(mc.expected_transitions(2))
[[0.085, 0.2975, 0.0,    0.0425]
 [0.0,   0.345,  0.1725, 0.0575]
 [0.0,   0.0,    0.7,    0.0   ]
 [0.15,  0.0,    0.15,   0.0   ]]

>>> print(mc.first_passage_probabilities(5, 3))
[[0.5, 0.0,    0.5,    0.0   ]
 [0.0, 0.35,   0.0,    0.05  ]
 [0.0, 0.07,   0.13,   0.045 ]
 [0.0, 0.0315, 0.1065, 0.03  ]
 [0.0, 0.0098, 0.0761, 0.0186]]
 
>>> print(mc.hitting_probabilities([0, 1]))
[1.0, 1.0, 0.0, 0.5]
 
>>> print(mc.walk(10))
['B', 'B', 'B', 'D', 'A', 'B', 'B', 'C', 'C', 'C']
```

Plotting functions can provide a visual representation of `MarkovChain` instances; in order to display function outputs immediately, the [interactive mode](https://matplotlib.org/stable/users/interactive.html#interactive-mode) of `Matplotlib` must be turned on:

```console
>>> plot_eigenvalues(mc)
```

![Eigenplot](https://i.imgur.com/ARWWG7z.png)

```console
>>> plot_graph(mc)
```

![Graphplot](https://i.imgur.com/looxKRO.png)

```console
>>> plot_walk(mc, 10, 'sequence')
```

![Walkplot](https://i.imgur.com/oxjDYr3.png)
