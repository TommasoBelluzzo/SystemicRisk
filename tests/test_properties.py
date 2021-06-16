# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

# Partial

from pydtmc import (
    MarkovChain
)

from pytest import (
    skip
)


#########
# TESTS #
#########

def test_attributes(p, is_absorbing, is_canonical, is_ergodic, is_reversible, is_symmetric):

    mc = MarkovChain(p)

    actual = mc.is_absorbing
    expected = is_absorbing

    assert actual == expected

    actual = mc.is_canonical
    expected = is_canonical

    assert actual == expected

    actual = mc.is_ergodic
    expected = is_ergodic

    assert actual == expected

    actual = mc.is_reversible
    expected = is_reversible

    assert actual == expected

    actual = mc.is_symmetric
    expected = is_symmetric

    assert actual == expected


def test_binary_matrices(p, accessibility_matrix, adjacency_matrix, communication_matrix):

    mc = MarkovChain(p)

    actual = mc.accessibility_matrix
    expected = np.asarray(accessibility_matrix)

    assert np.array_equal(actual, expected)

    for i in range(mc.size):
        for j in range(mc.size):

            actual = mc.is_accessible(j, i)
            expected = mc.accessibility_matrix[i, j] != 0
            assert actual == expected

            actual = mc.are_communicating(i, j)
            expected = mc.accessibility_matrix[i, j] != 0 and mc.accessibility_matrix[j, i] != 0
            assert actual == expected

    actual = mc.adjacency_matrix
    expected = np.asarray(adjacency_matrix)

    assert np.array_equal(actual, expected)

    actual = mc.communication_matrix
    expected = np.asarray(communication_matrix)

    assert np.array_equal(actual, expected)


def test_entropy(p, entropy_rate, entropy_rate_normalized, topological_entropy):

    mc = MarkovChain(p)

    actual = mc.entropy_rate
    expected = entropy_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.entropy_rate_normalized
    expected = entropy_rate_normalized

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.topological_entropy
    expected = topological_entropy

    assert np.isclose(actual, expected)


def test_fundamental_matrix(p, fundamental_matrix, kemeny_constant):

    mc = MarkovChain(p)

    actual = mc.fundamental_matrix
    expected = fundamental_matrix

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.kemeny_constant
    expected = kemeny_constant

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected


def test_irreducibility(p):

    mc = MarkovChain(p)

    if not mc.is_irreducible:
        skip('Markov chain is not irreducible.')
    else:

        actual = mc.states
        expected = mc.recurrent_states

        assert actual == expected

        actual = len(mc.communicating_classes)
        expected = 1

        assert actual == expected

        cf = mc.to_canonical_form()
        actual = cf.p
        expected = mc.p

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lumping_partitions(p, lumping_partitions):

    mc = MarkovChain(p)

    actual = mc.lumping_partitions
    expected = lumping_partitions

    assert actual == expected


def test_matrix(p, determinant, rank):

    mc = MarkovChain(p)

    actual = mc.determinant
    expected = determinant

    assert np.isclose(actual, expected)

    actual = mc.rank
    expected = rank

    assert actual == expected


def test_periodicity(p, period):

    mc = MarkovChain(p)

    actual = mc.period
    expected = period

    assert actual == expected

    actual = mc.is_aperiodic
    expected = period == 1

    assert actual == expected


def test_regularity(p):

    mc = MarkovChain(p)

    if not mc.is_regular:
        skip('Markov chain is not regular.')
    else:

        actual = mc.is_irreducible
        expected = True

        assert actual == expected

        values = np.sort(np.abs(npl.eigvals(mc.p)))
        actual = np.sum(np.logical_or(np.isclose(values, 1.0), values > 1.0))
        expected = 1

        assert actual == expected


def test_stationary_distributions(p, stationary_distributions):

    mc = MarkovChain(p)
    stationary_distributions = [np.array(stationary_distribution) for stationary_distribution in stationary_distributions]

    actual = len(mc.stationary_distributions)
    expected = len(stationary_distributions)

    assert actual == expected

    actual = len(mc.stationary_distributions)
    expected = len(mc.recurrent_classes)

    assert actual == expected

    ss_matrix = np.vstack(mc.stationary_distributions)
    actual = npl.matrix_rank(ss_matrix)
    expected = min(ss_matrix.shape)

    assert actual == expected

    for i in range(len(stationary_distributions)):

        assert np.isclose(np.sum(mc.steady_states[i]), 1.0)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = mc.steady_states[i]
        expected = stationary_distributions[i]

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_transitions(p):

    mc = MarkovChain(p)

    transition_matrix = mc.p
    states = mc.states

    for index, state in enumerate(states):

        actual = mc.conditional_probabilities(state)
        expected = transition_matrix[index, :]

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    for index1, state1 in enumerate(states):
        for index2, state2 in enumerate(states):

            actual = mc.transition_probability(state1, state2)
            expected = transition_matrix[index2, index1]

            assert np.isclose(actual, expected)


def test_times(p, mixing_rate, relaxation_rate, spectral_gap, implied_timescales):

    mc = MarkovChain(p)

    actual = mc.mixing_rate
    expected = mixing_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.relaxation_rate
    expected = relaxation_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.spectral_gap
    expected = spectral_gap

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.implied_timescales
    expected = implied_timescales

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
