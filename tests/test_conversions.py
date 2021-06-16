# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy.random as npr
import numpy.testing as npt

# Partial

from os import (
    close,
    remove
)

from pydtmc import (
    MarkovChain
)

from pytest import (
    mark
)

from random import (
    randint
)

from tempfile import (
    mkstemp
)


#########
# TESTS #
#########

def test_dictionary(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        d = mc_to.to_dictionary()
        mc_from = MarkovChain.from_dictionary(d)

        npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


@mark.slow
def test_graph(seed, maximum_size, runs):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        graph = mc_to.to_graph(False)
        mc_from = MarkovChain.from_graph(graph)

        npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)

        graph = mc_to.to_graph(True)
        mc_from = MarkovChain.from_graph(graph)

        npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


@mark.slow
def test_file(seed, maximum_size, runs, file_extension):

    for _ in range(runs):

        size = randint(2, maximum_size)
        zeros = randint(0, size)
        mc_to = MarkovChain.random(size, zeros=zeros, seed=seed)

        file_handler, file_path = mkstemp(suffix=file_extension)
        close(file_handler)

        # noinspection PyBroadException
        try:

            mc_to.to_file(file_path)
            mc_from = MarkovChain.from_file(file_path)

            exception = False

        except Exception:

            mc_from = None
            exception = True

            pass

        remove(file_path)

        assert exception is False
        npt.assert_allclose(mc_from.p, mc_to.p, rtol=1e-5, atol=1e-8)


def test_matrix(seed, maximum_size, runs):

    npr.seed(seed)

    for _ in range(runs):

        size = randint(2, maximum_size)

        m = npr.randint(101, size=(size, size))
        mc1 = MarkovChain.from_matrix(m)

        m = mc1.to_matrix()
        mc2 = MarkovChain.from_matrix(m)

        npt.assert_allclose(mc1.p, mc2.p, rtol=1e-5, atol=1e-8)
