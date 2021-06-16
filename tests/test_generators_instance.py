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

def test_bounded(p, boundary_condition, value):

    mc = MarkovChain(p)
    mc_bounded = mc.to_bounded_chain(boundary_condition)

    actual = mc_bounded.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_canonical(p, canonical_form):

    mc = MarkovChain(p)

    actual = mc.to_canonical_form().p

    if mc.is_canonical:
        expected = mc.p
    else:
        expected = np.asarray(canonical_form)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lazy(p, inertial_weights, value):

    mc = MarkovChain(p)
    mc_lazy = mc.to_lazy_chain(inertial_weights)

    actual = mc_lazy.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_lump(p, partitions, value):

    if value is None:
        skip('Markov chain is not lumpable for the specified partitions.')
    else:

        mc = MarkovChain(p)
        mc_lump = mc.lump(partitions)

        actual = mc_lump.p
        expected = np.asarray(value)

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_sub(p, states, value):

    if value is None:
        skip('Markov chain cannot generate the specified subchain.')
    else:

        mc = MarkovChain(p)

        try:

            mc_sub = mc.to_subchain(states)
            exception = False

        except ValueError:

            mc_sub = None
            exception = True

            pass

        assert exception is False

        actual = mc_sub.p
        expected = np.asarray(value)

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
