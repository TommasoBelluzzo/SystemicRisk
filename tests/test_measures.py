# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np
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

def test_absorption_probabilities(p, absorption_probabilities):

    mc = MarkovChain(p)

    actual = mc.absorption_probabilities()
    expected = absorption_probabilities

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_committor_probabilities(p, states1, states2, value_backward, value_forward):

    mc = MarkovChain(p)

    actual = mc.committor_probabilities('backward', states1, states2)
    expected = value_backward

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    actual = mc.committor_probabilities('forward', states1, states2)
    expected = value_forward

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_first_passage_probabilities(p, steps, initial_state, first_passage_states, value):

    mc = MarkovChain(p)

    actual = mc.first_passage_probabilities(steps, initial_state, first_passage_states)
    expected = np.asarray(value)

    if first_passage_states is not None:
        assert actual.size == steps

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_first_passage_reward(p, steps, initial_state, first_passage_states, rewards, value):

    mc = MarkovChain(p)

    if mc.size <= 2:
        skip('Markov chain size is less than or equal to 2.')
    else:

        actual = mc.first_passage_reward(steps, initial_state, first_passage_states, rewards)
        expected = value

        assert np.isclose(actual, expected)


def test_hitting_probabilities(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.hitting_probabilities(targets)
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if mc.is_irreducible:

        expected = np.ones(mc.size, dtype=float)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_hitting_times(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.hitting_times(targets)
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mean_first_passage_times_between(p, origins, targets, value):

    mc = MarkovChain(p)

    actual = mc.mean_first_passage_times_between(origins, targets)
    expected = value

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_mean_first_passage_times_to(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.mean_first_passage_times_to(targets)
    expected = value

    if actual is not None and expected is not None:

        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if targets is None:

            expected = np.dot(mc.p, expected) + np.ones((mc.size, mc.size), dtype=float) - np.diag(mc.mean_recurrence_times())
            npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    else:
        assert actual == expected


def test_mean_absorption_times(p, mean_absorption_times):

    mc = MarkovChain(p)

    actual = mc.mean_absorption_times()
    expected = mean_absorption_times

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_absorbing and len(mc.transient_states) > 0:

        actual = actual.size
        expected = mc.size - len(mc.absorbing_states)

        assert actual == expected


def test_mean_number_visits(p, mean_number_visits):

    mc = MarkovChain(p)

    actual = mc.mean_number_visits()
    expected = np.asarray(mean_number_visits)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mean_recurrence_times(p, mean_recurrence_times):

    mc = MarkovChain(p)

    actual = mc.mean_recurrence_times()
    expected = mean_recurrence_times

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected

    if mc.is_ergodic:

        actual = np.nan_to_num(actual**-1.0)
        expected = np.dot(actual, mc.p)

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_mixing_time(p, initial_distribution, jump, cutoff_type, value):

    mc = MarkovChain(p)

    actual = mc.mixing_time(initial_distribution, jump, cutoff_type)
    expected = value

    assert actual == expected


def test_sensitivity(p, state, value):

    mc = MarkovChain(p)

    actual = mc.sensitivity(state)
    expected = value

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_correlations(p, walk1, walk2, time_points, value):

    mc = MarkovChain(p)

    actual = np.asarray(mc.time_correlations(walk1, walk2, time_points))
    expected = value

    x = mc.walk(25)
    y = mc.walk(25)

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected


def test_time_relaxations(p, walk, initial_distribution, time_points, value):

    mc = MarkovChain(p)

    actual = np.asarray(mc.time_relaxations(walk, initial_distribution, time_points))
    expected = value

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    else:
        assert actual == expected
