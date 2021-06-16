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

def test_approximation(size, approximation_type, alpha, sigma, rho, k, value):

    mc = MarkovChain.approximation(size, approximation_type, alpha, sigma, rho, k)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_birth_death(p, q, value):

    mc = MarkovChain.birth_death(p, q)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_closest_reversible(p, distribution, weighted, value):

    mc = MarkovChain(p)
    cr = mc.closest_reversible(distribution, weighted)

    if mc.is_reversible:
        actual = cr.p
        expected = mc.p
    else:
        actual = cr.p
        expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_gamblers_ruin(size, w, value):

    mc = MarkovChain.gamblers_ruin(size, w)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_identity(size, value):

    mc = MarkovChain.identity(size)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_random(seed, size, zeros, mask, value):

    mc = MarkovChain.random(size, None, zeros, mask, seed)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if zeros > 0 and mask is None:

        actual = size**2 - np.count_nonzero(mc.p)
        expected = zeros

        assert actual == expected

    if mask is not None:

        indices = ~np.isnan(np.asarray(mask))

        actual = mc.p[indices]
        expected = np.asarray(value)[indices]

        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_urn_model(n, model, value):

    mc = MarkovChain.urn_model(n, model)

    actual = mc.p
    expected = np.asarray(value)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
