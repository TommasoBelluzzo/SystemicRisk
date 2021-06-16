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


#########
# TESTS #
#########

def test_classes_communicating(p, communicating_classes):

    mc = MarkovChain(p)

    actual = mc.communicating_classes
    expected = communicating_classes

    assert actual == expected

    if len(communicating_classes) > 1:

        npt.assert_array_equal(mc.communication_matrix, np.transpose(mc.communication_matrix))

    if np.array_equal(mc.p, np.eye(mc.size, dtype=float)):

        actual = [state for states in mc.communicating_classes for state in states]
        expected = mc.states

        assert actual == expected


def test_classes_recurrent(p, recurrent_classes):

    mc = MarkovChain(p)

    actual = mc.recurrent_classes
    expected = recurrent_classes

    assert actual == expected

    actual = sum([len(i) for i in mc.recurrent_classes])
    expected = len(set([state for recurrent_class in recurrent_classes for state in recurrent_class]))

    assert actual == expected


def test_classes_transient(p, transient_classes):

    mc = MarkovChain(p)

    actual = mc.transient_classes
    expected = transient_classes

    assert actual == expected

    actual = sum([len(i) for i in mc.transient_classes])
    expected = len(set([state for transient_class in transient_classes for state in transient_class]))

    assert actual == expected


def test_classes_cyclic(p, cyclic_classes):

    mc = MarkovChain(p)

    actual = mc.cyclic_classes
    expected = cyclic_classes

    assert actual == expected

    if np.array_equal(mc.p, np.eye(mc.size, dtype=float)):

        actual = mc.recurrent_states
        expected = mc.states

        assert actual == expected


def test_states_space(p, recurrent_classes, transient_classes):

    mc = MarkovChain(p)

    actual = mc.recurrent_states
    expected = sorted([state for recurrent_class in recurrent_classes for state in recurrent_class])

    assert actual == expected

    actual = mc.transient_states
    expected = sorted([state for transient_class in transient_classes for state in transient_class])

    assert actual == expected

    actual = sorted(mc.recurrent_states + mc.transient_states)
    expected = mc.states

    assert actual == expected

    if len(mc.recurrent_states) > 0:
        for state in mc.recurrent_states:
            assert mc.is_recurrent_state(state) is True

    if len(mc.transient_states) > 0:
        for state in mc.transient_states:
            assert mc.is_transient_state(state) is True


def test_states_absorbing(p, recurrent_classes, absorbing_states):

    mc = MarkovChain(p)

    actual = mc.absorbing_states
    expected = absorbing_states

    assert actual == expected

    actual = sum([1 if len(recurrent_class) == 1 and recurrent_class[0] in actual else 0 for recurrent_class in recurrent_classes])
    expected = len(absorbing_states)

    assert actual == expected

    if len(mc.absorbing_states) > 0:
        for state in mc.absorbing_states:
            assert mc.is_absorbing_state(state) is True


def test_states_cyclic(p):

    mc = MarkovChain(p)

    if len(mc.cyclic_states) > 0:
        for state in mc.cyclic_states:
            assert mc.is_cyclic_state(state) is True
