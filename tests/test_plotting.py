# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import matplotlib.pyplot as pp
import numpy.random as npr

# Partial

from pydtmc import (
    MarkovChain,
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)

from pytest import (
    mark
)

from random import (
    choice,
    getstate,
    randint,
    random,
    seed as setseed,
    setstate
)


#########
# TESTS #
#########

@mark.slow
def test_plot_eigenvalues(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, ax = plot_eigenvalues(mc)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False


@mark.slow
def test_plot_graph(seed, maximum_size, runs):

    rs = getstate()
    setseed(seed)

    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        configs.append((size, zeros) + tuple([random() < 0.5 for _ in range(4)]))

    setstate(rs)

    for i in range(runs):

        size, zeros, nodes_color, nodes_type, edges_color, edges_value = configs[i]

        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        # noinspection PyBroadException
        try:

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=True)
            pp.close(figure)

            figure, ax = plot_graph(mc, nodes_color=nodes_color, nodes_type=nodes_type, edges_color=edges_color, edges_value=edges_value, force_standard=False)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False


@mark.slow
def test_plot_redistributions(seed, maximum_size, maximum_distributions, runs):

    rs = getstate()
    setseed(seed)

    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        configs.append((size, zeros))

    setstate(rs)

    mcs = []
    plot_types = ['heatmap', 'projection']

    for i in range(runs):

        size, zeros = configs[i]
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        r = randint(1, maximum_distributions)

        distributions = r if random() < 0.5 else mc.redistribute(r, include_initial=True, output_last=False)
        initial_status = None if isinstance(distributions, int) or random() < 0.5 else distributions[0]
        plot_type = choice(plot_types)

        configs[i] = (distributions, initial_status, plot_type)
        mcs.append(mc)

    for i in range(runs):

        mc = mcs[i]
        distributions, initial_status, plot_type = configs[i]

        # noinspection PyBroadException
        try:

            figure, ax = plot_redistributions(mc, distributions, initial_status, plot_type)
            pp.close(figure)

            exception = False

        except Exception as ex:
            exception = True
            pass

        assert exception is False


@mark.slow
def test_plot_walk(seed, maximum_size, maximum_simulations, runs):

    rs = getstate()
    setseed(seed)

    configs = []

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)

        configs.append((size, zeros))

    setstate(rs)

    mcs = []
    plot_types = ['histogram', 'sequence', 'transitions']

    for i in range(runs):

        size, zeros = configs[i]
        mc = MarkovChain.random(size, zeros=zeros, seed=seed)

        r = randint(2, maximum_simulations)

        walk = r if random() < 0.5 else mc.walk(r, include_initial=True, output_indices=True)
        initial_state = None if isinstance(walk, int) or random() < 0.5 else walk[0]
        plot_type = choice(plot_types)

        configs[i] = (walk, initial_state, plot_type)
        mcs.append(mc)

    for i in range(runs):

        mc = mcs[i]
        walk, initial_state, plot_type = configs[i]

        # noinspection PyBroadException
        try:

            figure, ax = plot_walk(mc, walk, initial_state, plot_type)
            pp.close(figure)

            exception = False

        except Exception:
            exception = True
            pass

        assert exception is False
