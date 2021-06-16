# -*- coding: utf-8 -*-

__all__ = [
    'MarkovChain'
]


###########
# IMPORTS #
###########

# Full

import networkx as nx
import numpy as np
import numpy.linalg as npl
import scipy.stats as sps

# Partial

from copy import (
    deepcopy
)

from inspect import (
    getmembers,
    isfunction,
    stack,
    trace
)

from itertools import (
    chain
)

from math import (
    gamma,
    gcd,
    lgamma
)

# Internal

from .algorithms import *
from .base_class import *
from .computations import *
from .custom_types import *
from .decorators import *
from .exceptions import *
from .files_io import *
from .generators import *
from .measures import *
from .utilities import *
from .validation import *


###########
# CLASSES #
###########

@aliased
class MarkovChain(metaclass=BaseClass):

    """
    Defines a Markov chain with given transition matrix and state names.

    :param p: the transition matrix.
    :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
    :raises ValidationError: if any input argument is not compliant.
    """

    def __init__(self, p: tnumeric, states: olist_str = None):

        caller = stack()[1][3]
        sm = [x[1].__name__ for x in getmembers(MarkovChain, predicate=isfunction) if x[1].__name__[0] != '_' and isinstance(MarkovChain.__dict__.get(x[1].__name__), staticmethod)]

        if caller not in sm:

            try:

                p = validate_transition_matrix(p)

                if states is None:
                    states = [str(i) for i in range(1, p.shape[0] + 1)]
                else:
                    states = validate_state_names(states, p.shape[0])

            except Exception as e:  # pragma: no cover
                raise generate_validation_error(e, trace()) from None

        size = p.shape[0]

        graph = nx.DiGraph(p)
        graph = nx.relabel_nodes(graph, dict(zip(range(size), states)))

        self._cache: tcache = dict()
        self._digraph: tgraph = graph
        self._p: tarray = p
        self._size: int = size
        self._states: tlist_str = states

    def __eq__(self, other):

        if isinstance(other, MarkovChain):
            return np.array_equal(self.p, other.p) and self.states == other.states

        return NotImplemented

    def __hash__(self):

        return hash((self.p.tobytes(), tuple(self.states)))

    def __repr__(self) -> str:

        return self.__class__.__name__

    # noinspection PyListCreation
    def __str__(self) -> str:

        lines = []
        lines.append('')

        lines.append('DISCRETE-TIME MARKOV CHAIN')
        lines.append(f' SIZE:           {self._size:d}')
        lines.append(f' RANK:           {self.rank:d}')

        lines.append(f' CLASSES:        {len(self.communicating_classes):d}')
        lines.append(f'  > RECURRENT:   {len(self.recurrent_classes):d}')
        lines.append(f'  > TRANSIENT:   {len(self.transient_classes):d}')

        lines.append(f' ERGODIC:        {("YES" if self.is_ergodic else "NO")}')
        lines.append(f'  > APERIODIC:   {("YES" if self.is_aperiodic else "NO (" + str(self.period) + ")")}')
        lines.append(f'  > IRREDUCIBLE: {("YES" if self.is_irreducible else "NO")}')

        lines.append(f' ABSORBING:      {("YES" if self.is_absorbing else "NO")}')
        lines.append(f' REGULAR:        {("YES" if self.is_regular else "NO")}')
        lines.append(f' REVERSIBLE:     {("YES" if self.is_reversible else "NO")}')
        lines.append(f' SYMMETRIC:      {("YES" if self.is_symmetric else "NO")}')

        lines.append('')

        value = '\n'.join(lines)

        return value

    @cachedproperty
    def _absorbing_states_indices(self) -> tlist_int:

        indices = [index for index in range(self._size) if np.isclose(self._p[index, index], 1.0)]

        return indices

    @cachedproperty
    def _classes_indices(self) -> tlists_int:

        indices = [sorted([self._states.index(c) for c in scc]) for scc in nx.strongly_connected_components(self._digraph)]

        return indices

    @cachedproperty
    def _communicating_classes_indices(self) -> tlists_int:

        indices = sorted(self._classes_indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def _cyclic_classes_indices(self) -> tlists_int:

        if not self.is_irreducible:
            return list()

        if self.is_aperiodic:
            return self._communicating_classes_indices.copy()

        indices = find_cyclic_classes(self._p)
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def _cyclic_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self._cyclic_classes_indices)))

        return indices

    @cachedproperty
    def _eigenvalues_sorted(self) -> tarray:

        ev = eigenvalues_sorted(self._p)

        return ev

    @cachedproperty
    def _rdl_decomposition(self) -> trdl:

        r, d, l = rdl_decomposition(self._p)

        return r, d, l

    @cachedproperty
    def _recurrent_classes_indices(self) -> tlists_int:

        indices = [index for index in self._classes_indices if index not in self._transient_classes_indices]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def _recurrent_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self._recurrent_classes_indices)))

        return indices

    @cachedproperty
    def _slem(self) -> ofloat:

        if not self.is_ergodic:
            value = None
        else:
            value = slem(self._p)

        return value

    @cachedproperty
    def _states_indices(self) -> tlist_int:

        indices = list(range(self._size))

        return indices

    @cachedproperty
    def _transient_classes_indices(self) -> tlists_int:

        edges = set([edge1 for (edge1, edge2) in nx.condensation(self._digraph).edges])

        indices = [self._classes_indices[edge] for edge in edges]
        indices = sorted(indices, key=lambda x: (-len(x), x[0]))

        return indices

    @cachedproperty
    def _transient_states_indices(self) -> tlist_int:

        indices = sorted(list(chain.from_iterable(self._transient_classes_indices)))

        return indices

    @cachedproperty
    def absorbing_states(self) -> tlists_str:

        """
        A property representing the absorbing states of the Markov chain.
        """

        states = [*map(self._states.__getitem__, self._absorbing_states_indices)]

        return states

    @cachedproperty
    def accessibility_matrix(self) -> tarray:

        """
        A property representing the accessibility matrix of the Markov chain.
        """

        a = self.adjacency_matrix
        i = np.eye(self._size, dtype=int)

        am = (i + a)**(self._size - 1)
        am = (am > 0).astype(int)

        return am

    @cachedproperty
    def adjacency_matrix(self) -> tarray:

        """
        A property representing the adjacency matrix of the Markov chain.
        """

        am = (self._p > 0.0).astype(int)

        return am

    @cachedproperty
    def communicating_classes(self) -> tlists_str:

        """
        A property representing the communicating classes of the Markov chain.
        """

        classes = [[*map(self._states.__getitem__, i)] for i in self._communicating_classes_indices]

        return classes

    @cachedproperty
    def communication_matrix(self) -> tarray:

        """
        A property representing the communication matrix of the Markov chain.
        """

        cm = np.zeros((self._size, self._size), dtype=int)

        for index in self._communicating_classes_indices:
            cm[np.ix_(index, index)] = 1

        return cm

    @cachedproperty
    def cyclic_classes(self) -> tlists_str:

        """
        A property representing the cyclic classes of the Markov chain.
        """

        classes = [[*map(self._states.__getitem__, i)] for i in self._cyclic_classes_indices]

        return classes

    @cachedproperty
    def cyclic_states(self) -> tlists_str:

        """
        A property representing the cyclic states of the Markov chain.
        """

        states = [*map(self._states.__getitem__, self._cyclic_states_indices)]

        return states

    @cachedproperty
    def determinant(self) -> float:

        """
        A property representing the determinant of the transition matrix of the Markov chain.
        """

        d = npl.det(self._p)

        return d

    @cachedproperty
    def entropy_rate(self) -> ofloat:

        """
        A property representing the entropy rate of the Markov chain. If the Markov chain has multiple stationary distributions, then None is returned.
        """

        if len(self.pi) > 1:
            return None

        pi = self.pi[0]
        h = 0.0

        for i in range(self._size):
            for j in range(self._size):
                if self._p[i, j] > 0.0:
                    h += pi[i] * self._p[i, j] * np.log(self._p[i, j])

        if np.isclose(h, 0.0):
            return h

        return -h

    @cachedproperty
    def entropy_rate_normalized(self) -> ofloat:

        """
        A property representing the entropy rate, normalized between 0 and 1, of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        h = self.entropy_rate

        if h is None:
            return None

        if np.isclose(h, 0.0):
            hn = 0.0
        else:
            ev = eigenvalues_sorted(self.adjacency_matrix)
            hn = h / np.log(ev[-1])
            hn = min(1.0, max(0.0, hn))

        return hn

    @cachedproperty
    def fundamental_matrix(self) -> oarray:

        """
        A property representing the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing* or has no transient states, then None is returned.
        """

        if not self.is_absorbing or len(self.transient_states) == 0:
            return None

        indices = self._transient_states_indices

        q = self._p[np.ix_(indices, indices)]
        i = np.eye(len(indices), dtype=float)

        fm = npl.inv(i - q)

        return fm

    @cachedproperty
    def implied_timescales(self) -> oarray:

        """
        A property representing the implied timescales of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic:
            return None

        ev = self._eigenvalues_sorted[::-1]
        it = np.append(np.inf, -1.0 / np.log(ev[1:]))

        return it

    @cachedproperty
    def is_absorbing(self) -> bool:

        """
        A property indicating whether the Markov chain is absorbing.
        """

        if len(self.absorbing_states) == 0:
            return False

        indices = set(self._states_indices)
        absorbing_indices = set(self._absorbing_states_indices)
        transient_indices = set()

        progress = True
        unknown_states = None

        while progress:

            unknown_states = indices.copy() - absorbing_indices - transient_indices
            known_states = absorbing_indices | transient_indices

            progress = False

            for i in unknown_states:
                for j in known_states:
                    if self._p[i, j] > 0.0:
                        transient_indices.add(i)
                        progress = True
                        break

        result = len(unknown_states) == 0

        return result

    @cachedproperty
    def is_aperiodic(self) -> bool:

        """
        A property indicating whether the Markov chain is aperiodic.
        """

        if self.is_irreducible:
            result = set(self.periods).pop() == 1
        elif all(period == 1 for period in self.periods):
            result = True
        else:  # pragma: no cover
            result = nx.is_aperiodic(self._digraph)

        return result

    @cachedproperty
    def is_canonical(self) -> bool:

        """
        A property indicating whether the Markov chain has a canonical form.
        """

        recurrent_indices = self._recurrent_states_indices
        transient_indices = self._transient_states_indices

        if len(recurrent_indices) == 0 or len(transient_indices) == 0:
            return True

        result = max(transient_indices) < min(recurrent_indices)

        return result

    @cachedproperty
    def is_ergodic(self) -> bool:

        """
        A property indicating whether the Markov chain is ergodic or not.
        """

        result = self.is_irreducible and self.is_aperiodic

        return result

    @cachedproperty
    def is_irreducible(self) -> bool:

        """
        A property indicating whether the Markov chain is irreducible.
        """

        result = len(self.communicating_classes) == 1

        return result

    @cachedproperty
    def is_regular(self) -> bool:

        """
        A property indicating whether the Markov chain is regular.
        """

        d = np.diagonal(self._p)
        nz = np.count_nonzero(d)

        if nz > 0:
            k = (2 * self._size) - nz - 1
        else:
            k = self._size**self._size - (2 * self._size) + 2

        result = np.all(self._p**k > 0.0)

        return result

    @cachedproperty
    def is_reversible(self) -> bool:

        """
        A property indicating whether the Markov chain is reversible.
        """

        if len(self.pi) > 1:
            return False

        pi = self.pi[0]
        x = pi[:, np.newaxis] * self._p

        result = np.allclose(x, np.transpose(x))

        return result

    @cachedproperty
    def is_symmetric(self) -> bool:

        """
        A property indicating whether the Markov chain is symmetric.
        """

        result = np.allclose(self._p, np.transpose(self._p))

        return result

    @cachedproperty
    def kemeny_constant(self) -> ofloat:

        """
        A property representing the Kemeny's constant of the fundamental matrix of the Markov chain. If the Markov chain is not *absorbing* or has no transient states, then None is returned.
        """

        fm = self.fundamental_matrix

        if fm is None:
            return None

        if fm.size == 1:
            kc = fm[0].item()
        else:
            kc = np.trace(fm).item()

        return kc

    @cachedproperty
    def lumping_partitions(self) -> tparts:

        """
        A property representing all the partitions of the Markov chain that satisfy the ordinary lumpability criterion.
        """

        lp = find_lumping_partitions(self._p)

        return lp

    @cachedproperty
    def mixing_rate(self) -> ofloat:

        """
        A property representing the mixing rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self._slem is None:
            mr = None
        else:
            mr = -1.0 / np.log(self._slem)

        return mr

    @property
    def p(self) -> tarray:

        """
        A property representing the transition matrix of the Markov chain.
        """

        return self._p

    @cachedproperty
    def period(self) -> int:

        """
        A property representing the period of the Markov chain.
        """

        if self.is_aperiodic:
            return 1

        if self.is_irreducible:
            return set(self.periods).pop()

        period = 1

        for p in [self.periods[self.communicating_classes.index(recurrent_class)] for recurrent_class in self.recurrent_classes]:
            period = (period * p) // gcd(period, p)

        return period

    @cachedproperty
    def periods(self) -> tlist_int:

        """
        A property representing the period of each communicating class defined by the Markov chain.
        """

        periods = calculate_periods(self._digraph)

        return periods

    @alias('stationary_distributions', 'steady_states')
    @cachedproperty
    def pi(self) -> tlist_array:

        """
        A property representing the stationary distributions of the Markov chain.

        | **Aliases:** stationary_distributions, steady_states
        """

        if self.is_irreducible:
            s = np.reshape(gth_solve(self._p), (1, self._size))
        else:

            s = np.zeros((len(self.recurrent_classes), self._size), dtype=float)

            for i, indices in enumerate(self._recurrent_classes_indices):
                pr = self._p[np.ix_(indices, indices)]
                s[i, indices] = gth_solve(pr)

        pi = list()

        for i in range(s.shape[0]):
            pi.append(s[i, :])

        return pi

    @cachedproperty
    def rank(self) -> int:

        """
        A property representing the rank of the transition matrix of the Markov chain.
        """

        r = npl.matrix_rank(self._p)

        return r

    @cachedproperty
    def recurrent_classes(self) -> tlists_str:

        """
        A property representing the recurrent classes defined by the Markov chain.
        """

        classes = [[*map(self._states.__getitem__, i)] for i in self._recurrent_classes_indices]

        return classes

    @cachedproperty
    def recurrent_states(self) -> tlists_str:

        """
        A property representing the recurrent states of the Markov chain.
        """

        states = [*map(self._states.__getitem__, self._recurrent_states_indices)]

        return states

    @cachedproperty
    def relaxation_rate(self) -> ofloat:

        """
        A property representing the relaxation rate of the Markov chain. If the *SLEM* (second largest eigenvalue modulus) cannot be computed, then None is returned.
        """

        if self._slem is None:
            return None

        rr = 1.0 / (1.0 - self._slem)

        return rr

    @property
    def size(self) -> int:

        """
        A property representing the size of the Markov chain.
        """

        return self._size

    @cachedproperty
    def spectral_gap(self) -> ofloat:

        """
        A property representing the spectral gap of the Markov chain. If the Markov chain is not *ergodic*, then None is returned.
        """

        if not self.is_ergodic or self._slem is None:
            sg = None
        else:
            sg = 1.0 - self._slem

        return sg

    @property
    def states(self) -> tlist_str:

        """
        A property representing the states of the Markov chain.
        """

        return self._states

    @cachedproperty
    def topological_entropy(self) -> float:

        """
        A property representing the topological entropy of the Markov chain.
        """

        ev = eigenvalues_sorted(self.adjacency_matrix)
        te = np.log(ev[-1])

        return te

    @cachedproperty
    def transient_classes(self) -> tlists_str:

        """
        A property representing the transient classes defined by the Markov chain.
        """

        classes = [[*map(self._states.__getitem__, i)] for i in self._transient_classes_indices]

        return classes

    @cachedproperty
    def transient_states(self) -> tlists_str:

        """
        A property representing the transient states of the Markov chain.
        """

        states = [*map(self._states.__getitem__, self._transient_states_indices)]

        return states

    def absorption_probabilities(self) -> oarray:

        """
        A property representing the absorption probabilities of the Markov chain. If the Markov chain has no transient states, then None is returned.
        """

        if 'ap' not in self._cache:
            self._cache['ap'] = absorption_probabilities(self)

        return self._cache['ap']

    def are_communicating(self, state1: tstate, state2: tstate) -> bool:

        """
        The method verifies whether the given states of the Markov chain are communicating.

        :param state1: the first state.
        :param state2: the second state.
        :return: True if the given states are communicating, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state1 = validate_state(state1, self._states)
            state2 = validate_state(state2, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        a1 = self.accessibility_matrix[state1, state2] != 0
        a2 = self.accessibility_matrix[state2, state1] != 0
        result = a1 and a2

        return result

    def closest_reversible(self, distribution: onumeric = None, weighted: bool = False) -> tmc:

        """
        The method computes the closest reversible of the Markov chain.

        | **Notes:** the algorithm is described in `Computing the nearest reversible Markov chain (Nielsen & Weber, 2015) <http://doi.org/10.1002/nla.1967>`_.

        :param distribution: the distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param weighted: a boolean indicating whether to use the weighted Frobenius norm (by default, False).
        :return: a Markov chain representing the closest reversible.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the closest reversible could not be computed.
        """

        try:

            if distribution is None:
                distribution = np.ones(self._size, dtype=float) / self._size
            else:
                distribution = validate_vector(distribution, 'stochastic', False, size=self._size)

            weighted = validate_boolean(weighted)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        zeros = len(distribution) - np.count_nonzero(distribution)

        if weighted and zeros > 0:  # pragma: no cover
            raise ValidationError('If the weighted Frobenius norm is used, the distribution must not contain zero-valued probabilities.')

        if self.is_reversible:
            p = np.copy(self._p)
        else:

            p, error_message = closest_reversible(self._p, distribution, weighted)

            if error_message is not None:  # pragma: no cover
                raise ValueError(error_message)

        mc = MarkovChain(p, self._states)

        if not mc.is_reversible:  # pragma: no cover
            raise ValueError('The closest reversible could not be computed.')

        return mc

    def committor_probabilities(self, committor_type: str, states1: tstates, states2: tstates) -> oarray:

        """
        The method computes the committor probabilities between the given subsets of the state space defined by the Markov chain.

        :param committor_type: the type of committor whose probabilities must be computed (either backward or forward).
        :param states1: the first subset of the state space.
        :param states2: the second subset of the state space.
        :return: the committor probabilities if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            committor_type = validate_enumerator(committor_type, ['backward', 'forward'])
            states1 = validate_states(states1, self._states, 'subset', True)
            states2 = validate_states(states2, self._states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        intersection = np.intersect1d(states1, states2)

        if len(intersection) > 0:  # pragma: no cover
            raise ValidationError(f'The two sets of states must be disjoint. An intersection has been detected: {", ".join([str(i) for i in intersection])}.')

        value = committor_probabilities(self, committor_type, states1, states2)

        return value

    @alias('conditional_distribution')
    def conditional_probabilities(self, state: tstate) -> tarray:

        """
        The method computes the probabilities, for all the states of the Markov chain, conditioned on the process being at a given state.

        | **Aliases:** conditional_distribution

        :param state: the current state.
        :return: the conditional probabilities of the Markov chain states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = self._p[state, :]

        return value

    def expected_rewards(self, steps: int, rewards: tnumeric) -> tarray:

        """
        The method computes the expected rewards of the Markov chain after *N* steps, given the reward value of each state.

        :param steps: the number of steps.
        :param rewards: the reward values.
        :return: the expected rewards of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rewards = validate_rewards(rewards, self._size)
            steps = validate_integer(steps, lower_limit=(0, True))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        original_rewards = np.copy(rewards)
        value = np.copy(rewards)

        for _ in range(steps):
            value = original_rewards + np.dot(value, self._p)

        return value

    def expected_transitions(self, steps: int, initial_distribution: onumeric = None) -> oarray:

        """
        The method computes the expected number of transitions performed by the Markov chain after *N* steps, given the initial distribution of the states.

        :param steps: the number of steps.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :return: the expected number of transitions on each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if steps <= self._size:

            pi = initial_distribution
            p_sum = initial_distribution

            for _ in range(steps - 1):
                pi = np.dot(pi, self._p)
                p_sum += pi

            et = p_sum[:, np.newaxis] * self._p

        else:

            r, d, l = self._rdl_decomposition(self._p)
            q = np.asarray(np.diagonal(d))

            if q.size == 1:
                q = q.item()
                gs = steps if np.isclose(q, 1.0) else (1.0 - q**steps) / (1.0 - q)
            else:
                gs = np.zeros(np.shape(q), dtype=q.dtype)
                indices = (q == 1.0)
                gs[indices] = steps
                gs[~indices] = (1.0 - q[~indices]**steps) / (1.0 - q[~indices])

            ds = np.diag(gs)
            ts = np.dot(np.dot(r, ds), np.conjugate(l))
            ps = np.dot(initial_distribution, ts)

            et = np.real(ps[:, np.newaxis] * self._p)

        return et

    @alias('fpp')
    def first_passage_probabilities(self, steps: int, initial_state: tstate, first_passage_states: ostates = None) -> tarray:

        """
        The method computes the first passage probabilities of the Markov chain after *N* steps, given an initial state and, optionally, the first passage states.

        | **Aliases:** fpp

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :return: the first passage probabilities of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))
            initial_state = validate_state(initial_state, self._states)

            if first_passage_states is not None:
                first_passage_states = validate_states(first_passage_states, self._states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = first_passage_probabilities(self, steps, initial_state, first_passage_states)

        return value

    @alias('fpt')
    def first_passage_reward(self, steps: int, initial_state: tstate, first_passage_states: tstates, rewards: tnumeric) -> float:

        """
        The method computes the first passage reward of the Markov chain after *N* steps, given the reward value of each state, the initial state and the first passage states.

        | **Aliases:** fpt

        :param steps: the number of steps.
        :param initial_state: the initial state.
        :param first_passage_states: the first passage states.
        :param rewards: the reward values.
        :return: the first passage reward of the Markov chain for the given configuration.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states.
        """

        try:

            initial_state = validate_state(initial_state, self._states)
            first_passage_states = validate_states(first_passage_states, self._states, 'subset', True)
            rewards = validate_rewards(rewards, self._size)
            steps = validate_integer(steps, lower_limit=(0, True))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if self._size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states and the first passage rewards cannot be computed.')

        if initial_state in first_passage_states:  # pragma: no cover
            raise ValidationError('The first passage states cannot include the initial state.')

        if len(first_passage_states) == (self._size - 1):  # pragma: no cover
            raise ValidationError('The first passage states cannot include all the states except the initial one.')

        value = first_passage_reward(self, steps, initial_state, first_passage_states, rewards)

        return value

    def hitting_probabilities(self, targets: ostates = None) -> tarray:

        """
        The method computes the hitting probability, for the states of the Markov chain, to the given set of states.

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the hitting probability of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is None:
                targets = self._states_indices.copy()
            else:
                targets = validate_states(targets, self._states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = hitting_probabilities(self, targets)

        return value

    def hitting_times(self, targets: ostates = None) -> tarray:

        """
        The method computes the hitting times, for all the states of the Markov chain, to the given set of states.

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the hitting probability of each state of the Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is None:
                targets = self._states_indices.copy()
            else:
                targets = validate_states(targets, self._states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = hitting_times(self, targets)

        return value

    def is_absorbing_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state of the Markov chain is absorbing.

        :param state: the target state.
        :return: True if the state is absorbing, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self._absorbing_states_indices

        return result

    def is_accessible(self, state_target: tstate, state_origin: tstate) -> bool:

        """
        The method verifies whether the given target state is reachable from the given origin state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: True if the target state is reachable from the origin state, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self._states)
            state_origin = validate_state(state_origin, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = self.accessibility_matrix[state_origin, state_target] != 0

        return result

    def is_cyclic_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is cyclic.

        :param state: the target state.
        :return: True if the state is cyclic, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self._cyclic_states_indices

        return result

    def is_recurrent_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is recurrent.

        :param state: the target state.
        :return: True if the state is recurrent, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self._recurrent_states_indices

        return result

    def is_transient_state(self, state: tstate) -> bool:

        """
        The method verifies whether the given state is transient.

        :param state: the target state.
        :return: True if the state is transient, False otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        result = state in self._transient_states_indices

        return result

    def lump(self, partitions: tpart) -> tmc:

        """
        The method attempts to reduce the state space of the Markov chain with respect to the given partitions following the ordinary lumpability criterion.

        :param partitions: the partitions of the state space.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Markov chain defines only two states or is not strongly lumpable with respect to the given partitions.
        """

        try:

            partitions = validate_partitions(partitions, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if self._size == 2:  # pragma: no cover
            raise ValueError('The Markov chain defines only two states and cannot be lumped.')

        p, states, error_message = lump(self.p, self.states, partitions)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @alias('mat')
    def mean_absorption_times(self) -> oarray:

        """
        The method computes the mean absorption times of the Markov chain.

        | **Aliases:** mat

        :return: the mean absorption times if the Markov chain is *absorbing* or has transient states, None otherwise.
        """

        if 'mat' not in self._cache:
            self._cache['mat'] = mean_absorption_times(self)

        return self._cache['mat']

    @alias('mfpt_between', 'mfptb')
    def mean_first_passage_times_between(self, origins: tstates, targets: tstates) -> ofloat:

        """
        The method computes the mean first passage times between the given subsets of the state space.

        | **Aliases:** mfpt_between, mfptb

        :param origins: the origin states.
        :param targets: the target states.
        :return: the mean first passage times between the given subsets of the state space if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            origins = validate_states(origins, self._states, 'subset', True)
            targets = validate_states(targets, self._states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = mean_first_passage_times_between(self, origins, targets)

        return value

    @alias('mfpt_to', 'mfptt')
    def mean_first_passage_times_to(self, targets: ostates = None) -> oarray:

        """
        The method computes the mean first passage times, for all the states, to the given set of states.

        | **Aliases:** mfpt_to, mfptt

        :param targets: the target states (if omitted, all the states are targeted).
        :return: the mean first passage times to targeted states if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if targets is not None:
                targets = validate_states(targets, self._states, 'regular', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = mean_first_passage_times_to(self, targets)

        return value

    @alias('mnv')
    def mean_number_visits(self) -> oarray:

        """
        The method computes the mean number of visits of the Markov chain.

        | **Aliases:** mnv

        :return: the mean number of visits.
        """

        if 'mnv' not in self._cache:
            self._cache['mnv'] = mean_number_visits(self)

        return self._cache['mnv']

    @alias('mrt')
    def mean_recurrence_times(self) -> oarray:

        """
        The method computes the mean recurrence times of the Markov chain.

        | **Aliases:** mrt

        :return: the mean recurrence times if the Markov chain is *ergodic*, None otherwise.
        """

        if 'mrt' not in self._cache:
            self._cache['mrt'] = mean_recurrence_times(self)

        return self._cache['mrt']

    def mixing_time(self, initial_distribution: onumeric = None, jump: int = 1, cutoff_type: str = 'natural') -> oint:

        """
        The method computes the mixing time of the Markov chain, given the initial distribution of the states.

        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param jump: the number of steps in each iteration (by default, 1).
        :param cutoff_type: the type of cutoff to use (either natural or traditional; by default, natural).
        :return: the mixing time if the Markov chain is *ergodic*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

            jump = validate_integer(jump, lower_limit=(0, True))
            cutoff_type = validate_enumerator(cutoff_type, ['natural', 'traditional'])

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if cutoff_type == 'traditional':
            cutoff = 0.25
        else:
            cutoff = 1.0 / (2.0 * np.exp(1.0))

        value = mixing_time(self, initial_distribution, jump, cutoff)

        return value

    def predict(self, steps: int, initial_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> twalk:

        """
        The method simulates the most probable outcome in a random walk of *N* steps.

        | **Notes:** in case of probability tie, the subsequent state is chosen uniformly at random among all the equiprobable states.

        :param steps: the number of steps.
        :param initial_state: the initial state of the prediction (if omitted, it is chosen uniformly at random).
        :param include_initial: a boolean indicating whether to include the initial state in the output sequence (by default, False).
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the sequence of states produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)
            output_indices = validate_boolean(output_indices)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        prediction = list()

        if include_initial:
            prediction.append(initial_state)

        current_state = initial_state

        for _ in range(steps):
            d = self._p[current_state, :]
            d_max = np.argwhere(d == np.max(d))

            w = np.zeros(self._size, dtype=float)
            w[d_max] = 1.0 / d_max.size

            current_state = rng.choice(self._size, size=1, p=w).item()
            prediction.append(current_state)

        if not output_indices:
            prediction = [*map(self._states.__getitem__, prediction)]

        return prediction

    def prior_probabilities(self, hyperparameter: onumeric = None) -> tarray:

        """
        The method computes the prior probabilities, in logarithmic form, of the Markov chain.

        :param hyperparameter: the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            if hyperparameter is None:
                hyperparameter = np.ones((self._size, self._size), dtype=float)
            else:
                hyperparameter = validate_hyperparameter(hyperparameter, self._size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        lps = np.zeros(self._size, dtype=float)

        for i in range(self._size):

            lp = 0.0

            for j in range(self._size):
                hij = hyperparameter[i, j]
                lp += (hij - 1.0) * np.log(self._p[i, j]) - lgamma(hij)

            lps[i] = (lp + lgamma(np.sum(hyperparameter[i, :])))

        return lps

    def redistribute(self, steps: int, initial_status: ostatus = None, include_initial: bool = False, output_last: bool = True) -> tlist_array:

        """
        The method simulates a redistribution of states of *N* steps.

        :param steps: the number of steps.
        :param initial_status: the initial state or the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param include_initial: a boolean indicating whether to include the initial distribution in the output sequence (by default, False).
        :param output_last: a boolean indicating whether to the output only the last distributions (by default, True).
        :return: the sequence of redistributions produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            steps = validate_integer(steps, lower_limit=(0, True))

            if initial_status is None:
                initial_status = np.ones(self._size, dtype=float) / self._size
            else:
                initial_status = validate_status(initial_status, self._states)

            include_initial = validate_boolean(include_initial)
            output_last = validate_boolean(output_last)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        distributions = np.zeros((steps, self._size), dtype=float)

        for i in range(steps):

            if i == 0:
                distributions[i, :] = initial_status.dot(self._p)
            else:
                distributions[i, :] = distributions[i - 1, :].dot(self._p)

            distributions[i, :] /= np.sum(distributions[i, :])

        if output_last:
            distributions = distributions[-1:, :]

        if include_initial:
            distributions = np.vstack((initial_status, distributions))

        return [np.ravel(x) for x in np.split(distributions, distributions.shape[0])]

    def sensitivity(self, state: tstate) -> oarray:

        """
        The method computes the sensitivity matrix of the stationary distribution with respect to a given state.

        :param state: the target state.
        :return: the sensitivity matrix of the stationary distribution if the Markov chain is *irreducible*, None otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state = validate_state(state, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = sensitivity(self, state)

        return value

    def time_correlations(self, walk1: twalk, walk2: owalk = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time autocorrelations of a single observed sequence of states or the time cross-correlations of two observed sequences of states.

        :param walk1: the first observed sequence of states.
        :param walk2: the second observed sequence of states.
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain has multiple stationary distributions, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk1 = validate_states(walk1, self._states, 'walk', False)

            if walk2 is not None:
                walk2 = validate_states(walk2, self._states, 'walk', False)

            time_points = validate_time_points(time_points)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = time_correlations(self, self._rdl_decomposition, walk1, walk2, time_points)

        return value

    def time_relaxations(self, walk: twalk, initial_distribution: onumeric = None, time_points: ttimes_in = 1) -> otimes_out:

        """
        The method computes the time relaxations of an observed sequence of states with respect to the given initial distribution of the states.

        :param walk: the observed sequence of states.
        :param initial_distribution: the initial distribution of the states (if omitted, the states are assumed to be uniformly distributed).
        :param time_points: the time point or a list of time points at which the computation is performed (by default, 1).
        :return: None if the Markov chain has multiple stationary distributions, a float value if *time_points* is provided as an integer, a list of float values otherwise.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self._states, 'walk', False)

            if initial_distribution is None:
                initial_distribution = np.ones(self._size, dtype=float) / self._size
            else:
                initial_distribution = validate_vector(initial_distribution, 'stochastic', False, size=self._size)

            time_points = validate_time_points(time_points)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = time_relaxations(self, self._rdl_decomposition, walk, initial_distribution, time_points)

        return value

    @alias('to_bounded')
    def to_bounded_chain(self, boundary_condition: tbcond) -> tmc:

        """
        The method returns a bounded Markov chain by adjusting the transition matrix of the original process using the specified boundary condition.

        | **Aliases:** to_bounded

        :param boundary_condition:
         - a float representing the first probability of the semi-reflecting condition;
         - a string representing the boundary condition type (either absorbing or reflecting).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            boundary_condition = validate_boundary_condition(boundary_condition)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = bounded(self._p, boundary_condition)
        mc = MarkovChain(p, self._states)

        return mc

    @alias('to_canonical')
    def to_canonical_form(self) -> tmc:

        """
        The method returns the canonical form of the Markov chain.

        | **Aliases:** to_canonical

        :return: a Markov chain.
        """

        p, _ = canonical(self._p, self._recurrent_states_indices, self._transient_states_indices)
        states = [*map(self._states.__getitem__, self._transient_states_indices + self._recurrent_states_indices)]
        mc = MarkovChain(p, states)

        return mc

    def to_dictionary(self) -> tmc_dict:

        """
        The method returns a dictionary representing the Markov chain transitions.

        :return: a dictionary.
        """

        d = {}

        for i in range(self._size):
            for j in range(self._size):
                d[(self._states[i], self._states[j])] = self._p[i, j]

        return d

    def to_graph(self, multi: bool = False) -> tgraphs:

        """
        The method returns a directed graph representing the Markov chain.

        :param multi: a boolean indicating whether the graph is allowed to define multiple edges between two nodes (by default, False).
        :return: a directed graph.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            multi = validate_boolean(multi)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if multi:
            graph = nx.MultiDiGraph(self._p)
            graph = nx.relabel_nodes(graph, dict(zip(range(self._size), self._states)))
        else:
            graph = deepcopy(self._digraph)

        return graph

    def to_file(self, file_path: str):

        """
        The method writes a Markov chain to the given file.

        | Only csv, json, xml and plain text files are supported; data format is inferred from the file extension.

        :param file_path: the location of the file in which the Markov chain must be written.
        :raises OSError: if the file cannot be written.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            file_path = validate_string(file_path)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        file_extension = get_file_extension(file_path)

        if file_extension not in ['.csv', '.json', '.txt', '.xml']:  # pragma: no cover
            raise ValidationError('Only csv, json, xml and plain text files are supported.')

        d = self.to_dictionary()

        if file_extension == '.csv':
            write_csv(d, file_path)
        elif file_extension == '.json':
            write_json(d, file_path)
        elif file_extension == '.txt':
            write_txt(d, file_path)
        else:
            write_xml(d, file_path)

    @alias('to_lazy')
    def to_lazy_chain(self, inertial_weights: tweights = 0.5) -> tmc:

        """
        The method returns a lazy Markov chain by adjusting the state inertia of the original process.

        | **Aliases:** to_lazy

        :param inertial_weights: the inertial weights to apply for the transformation (by default, 0.5).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            inertial_weights = validate_vector(inertial_weights, 'unconstrained', True, size=self._size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = lazy(self._p, inertial_weights)
        mc = MarkovChain(p, self._states)

        return mc

    def to_matrix(self) -> tarray:

        """
        The method returns a transition matrix representing the Markov chain.

        :return: a transition matrix.
        """

        m = np.copy(self._p)

        return m

    def to_subchain(self, states: tstates) -> tmc:

        """
        The method returns a subchain containing all the given states plus all the states reachable from them.

        :param states: the states to include in the subchain.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the subchain is not a valid Markov chain.
        """

        try:

            states = validate_states(states, self._states, 'subset', True)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, error_message = sub(self._p, self._states, self.adjacency_matrix, states)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    def transition_probability(self, state_target: tstate, state_origin: tstate) -> float:

        """
        The method computes the probability of a given state, conditioned on the process being at a given specific state.

        :param state_target: the target state.
        :param state_origin: the origin state.
        :return: the transition probability of the target state.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            state_target = validate_state(state_target, self._states)
            state_origin = validate_state(state_origin, self._states)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        value = self._p[state_origin, state_target]

        return value

    def walk(self, steps: int, initial_state: ostate = None, final_state: ostate = None, include_initial: bool = False, output_indices: bool = False, seed: oint = None) -> twalk:

        """
        The method simulates a random walk of *N* steps.

        :param steps: the number of steps.
        :param initial_state: the initial state of the walk (if omitted, it is chosen uniformly at random).
        :param final_state: the final state of the walk (if specified, the simulation stops as soon as it is reached even if not all the steps have been performed).
        :param include_initial: a boolean indicating whether to include the initial state in the output sequence (by default, False).
        :param output_indices: a boolean indicating whether to the output the state indices (by default, False).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: the sequence of states produced by the simulation.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)
            steps = validate_integer(steps, lower_limit=(1, False))

            if initial_state is None:
                initial_state = rng.randint(0, self._size)
            else:
                initial_state = validate_state(initial_state, self._states)

            include_initial = validate_boolean(include_initial)

            if final_state is not None:
                final_state = validate_state(final_state, self._states)

            output_indices = validate_boolean(output_indices)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        walk = list()

        if include_initial:
            walk.append(initial_state)

        current_state = initial_state

        for _ in range(steps):

            w = self._p[current_state, :]
            current_state = rng.choice(self._size, size=1, p=w).item()
            walk.append(current_state)

            if current_state == final_state:
                break

        if not output_indices:
            walk = [*map(self._states.__getitem__, walk)]

        return walk

    def walk_probability(self, walk: twalk) -> float:

        """
        The method computes the probability of a given sequence of states.

        :param walk: the observed sequence of states.
        :return: the probability of the sequence of states.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            walk = validate_states(walk, self._states, 'walk', False)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p = 0.0

        for (i, j) in zip(walk[:-1], walk[1:]):
            if self._p[i, j] > 0.0:
                p += np.log(self._p[i, j])
            else:
                p = -np.inf
                break

        return np.exp(p)

    @staticmethod
    def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: ofloat = None) -> tmc:

        """
        The method approximates the Markov chain associated with the discretized version of the following first-order autoregressive process:

        | :math:`y_t = (1 - \\rho) \\alpha + \\rho y_{t-1} + \\varepsilon_t`
        | with :math:`\\varepsilon_t \\overset{i.i.d}{\\sim} \\mathcal{N}(0, \\sigma_{\\varepsilon}^{2})`

        :param size: the size of the Markov chain.
        :param approximation_type:
         - *adda-cooper* for the Adda-Cooper approximation;
         - *rouwenhorst* for the Rouwenhorst approximation;
         - *tauchen* for the Tauchen approximation;
         - *tauchen-hussey* for the Tauchen-Hussey approximation.
        :param alpha: the constant term :math:`\\alpha`, representing the unconditional mean of the process.
        :param sigma: the standard deviation of the innovation term :math:`\\varepsilon`.
        :param rho: the autocorrelation coefficient :math:`\\rho`, representing the persistence of the process across periods.
        :param k:
         - in the Tauchen approximation, the number of standard deviations to approximate out to (if omitted, the value is set to 3);
         - in the Tauchen-Hussey approximation, the standard deviation used for the gaussian quadrature (if omitted, the value is set to an optimal default).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the gaussian quadrature fails to converge in the Tauchen-Hussey approach.
        """

        try:

            size = validate_integer(size, lower_limit=(2, False))
            approximation_type = validate_enumerator(approximation_type, ['adda-cooper', 'rouwenhorst', 'tauchen', 'tauchen-hussey'])
            alpha = validate_float(alpha)
            sigma = validate_float(sigma, lower_limit=(0.0, True))
            rho = validate_float(rho, lower_limit=(-1.0, False), upper_limit=(1.0, False))

            if approximation_type == 'tauchen':
                if k is None:
                    k = 3.0
                else:
                    k = validate_float(k, lower_limit=(1.0, False))
            elif approximation_type == 'tauchen-hussey':
                if k is None:
                    w = 0.5 + (rho / 4.0)
                    k = (w * sigma) + ((1 - w) * (sigma / np.sqrt(1.0 - rho ** 2.0)))
                else:
                    k = validate_float(k, lower_limit=(0.0, True))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, error_message = approximation(size, approximation_type, alpha, sigma, rho, k)

        if error_message is not None:  # pragma: no cover
            raise ValueError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def birth_death(p: tarray, q: tarray, states: olist_str = None) -> tmc:

        """
        The method generates a birth-death Markov chain of given size and from given probabilities.

        :param q: the creation probabilities.
        :param p: the annihilation probabilities.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            p = validate_vector(p, 'creation', False)
            q = validate_vector(q, 'annihilation', False)

            if states is None:
                states = [str(i) for i in range(1, {p.shape[0], q.shape[0]}.pop() + 1)]
            else:
                states = validate_state_names(states, {p.shape[0], q.shape[0]}.pop())

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        if p.shape[0] != q.shape[0]:  # pragma: no cover
            raise ValidationError('The vector of annihilation probabilities and the vector of creation probabilities must have the same size.')

        if not np.all(q + p <= 1.0):  # pragma: no cover
            raise ValidationError('The sums of annihilation and creation probabilities must be less than or equal to 1.')

        p, _ = birth_death(p, q)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def fit_function(possible_states: tlist_str, f: ttfunc, quadrature_type: str, quadrature_interval: ointerval = None) -> tmc:

        """
        The method fits a Markov chain using the given transition function and the given quadrature type for the computation of nodes and weights.

        :param possible_states: the possible states of the process.
        :param f: the transition function of the process.
        :param quadrature_type:
         - *gauss-chebyshev* for the Gauss-Chebyshev quadrature;
         - *gauss-legendre* for the Gauss-Legendre quadrature;
         - *niederreiter* for the Niederreiter equidistributed sequence;
         - *newton-cotes* for the Newton-Cotes quadrature;
         - *simpson-rule* for the Simpson rule;
         - *trapezoid-rule* for the Trapezoid rule.
        :param quadrature_interval: the quadrature interval to use for the computation of nodes and weights (by default, the interval [0, 1]).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the Gauss-Legendre quadrature fails to converge.
        """

        try:

            possible_states = validate_state_names(possible_states)
            f = validate_transition_function(f)
            quadrature_type = validate_enumerator(quadrature_type, ['gauss-chebyshev', 'gauss-legendre', 'niederreiter', 'newton-cotes', 'simpson-rule', 'trapezoid-rule'])

            if quadrature_interval is None:
                quadrature_interval = (0.0, 1.0)
            else:
                quadrature_interval = validate_interval(quadrature_interval)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        size = len(possible_states)

        a = quadrature_interval[0]
        b = quadrature_interval[1]

        if quadrature_type == 'gauss-chebyshev':

            t1 = np.arange(size) + 0.5
            t2 = np.arange(0.0, size, 2.0)
            t3 = np.concatenate((np.array([1.0]), -2.0 / (np.arange(1.0, size - 1.0, 2) * np.arange(3.0, size + 1.0, 2))))

            nodes = ((b + a) / 2.0) - ((b - a) / 2.0) * np.cos((np.pi / size) * t1)
            weights = ((b - a) / size) * np.cos((np.pi / size) * np.outer(t1, t2)) @ t3

        elif quadrature_type == 'gauss-legendre':

            nodes = np.zeros(size, dtype=float)
            weights = np.zeros(size, dtype=float)

            iterations = 0
            i = np.arange(int(np.fix((size + 1.0) / 2.0)))
            pp = 0.0
            z = np.cos(np.pi * ((i + 1.0) - 0.25) / (size + 0.5))

            while iterations < 100:

                iterations += 1

                p1 = np.ones_like(z, dtype=float)
                p2 = np.zeros_like(z, dtype=float)

                for j in range(1, size + 1):
                    p3 = p2
                    p2 = p1
                    p1 = ((((2.0 * j) - 1.0) * z * p2) - ((j - 1) * p3)) / j

                pp = size * (((z * p1) - p2) / (z**2.0 - 1.0))

                z1 = np.copy(z)
                z = z1 - (p1 / pp)

                if np.allclose(abs(z - z1), 0.0):
                    break

            if iterations == 100:
                raise ValueError('The Gauss-Legendre quadrature failed to converge.')

            xl = 0.5 * (b - a)
            xm = 0.5 * (b + a)

            nodes[i] = xm - (xl * z)
            nodes[-i - 1] = xm + (xl * z)

            weights[i] = (2.0 * xl) / ((1.0 - z**2.0) * pp**2.0)
            weights[-i - 1] = weights[i]

        elif quadrature_type == 'niederreiter':

            r = b - a

            nodes = np.arange(1.0, size + 1.0) * 2.0**0.5
            nodes -= np.fix(nodes)
            nodes = a + (nodes * r)

            weights = (r / size) * np.ones(size, dtype=float)

        elif quadrature_type == 'simpson-rule':

            if (size % 2) == 0:
                raise ValidationError('The Simpson quadrature requires an odd number of possible states.')

            nodes = np.linspace(a, b, size)

            weights = np.kron(np.ones((size + 1) // 2, dtype=float), np.array([2.0, 4.0]))
            weights = weights[:size]
            weights[0] = weights[-1] = 1
            weights = ((nodes[1] - nodes[0]) / 3.0) * weights

        elif quadrature_type == 'trapezoid-rule':

            nodes = np.linspace(a, b, size)

            weights = (nodes[1] - nodes[0]) * np.ones(size)
            weights[0] *= 0.5
            weights[-1] *= 0.5

        else:

            bandwidth = (b - a) / size

            nodes = (np.arange(size) + 0.5) * bandwidth
            weights = np.repeat(bandwidth, size)

        p = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                p[i, j] = f(nodes[i], nodes[j]) * weights[j]

        for i in range(p.shape[0]):
            p[i, :] /= np.sum(p[i, :])

        return MarkovChain(p, possible_states)

    @staticmethod
    def fit_walk(fitting_type: str, possible_states: tlist_str, walk: twalk, k: tany = None, confidence_level: float = 0.95) -> tmc_fit:

        """
        The method fits a Markov chain from an observed sequence of states using the specified approach and computes the multinomial confidence intervals of the fitting.

        | **Notes:** the algorithm for the computation of multinomial confidence intervals is described in `Constructing two-sided simultaneous confidence intervals for multinomial proportions (May & Johnson, 2000) <http://dx.doi.org/10.18637/jss.v005.i06>`_.

        :param fitting_type:
         - *map* for the maximum a posteriori approach;
         - *mle* for the maximum likelihood approach.
        :param possible_states: the possible states of the process.
        :param walk: the observed sequence of states.
        :param k:
         - in the maximum a posteriori approach, the matrix for the a priori distribution (if omitted, a default value of 1 is assigned to each parameter);
         - in the maximum likelihood approach, a boolean indicating whether to apply a Laplace smoothing to compensate for the unseen transition combinations (if omitted, the value is set to False).
        :param confidence_level: the confidence level used for the computation of the multinomial confidence intervals (by default, 0.95).
        :return: a tuple whose first element is a Markov chain and whose second element represents the multinomial confidence intervals of the fitting (0: lower, 1: upper).
        :raises ValidationError: if any input argument is not compliant.
        """

        def compute_moments(cm_c: int, cm_xi: int) -> tarray:

            cm_a = cm_xi + cm_c
            cm_b = max(0, cm_xi - cm_c)

            if cm_b > 0:
                d = sps.poisson.cdf(cm_a, cm_xi) - sps.poisson.cdf(cm_b - 1, cm_xi)
            else:
                d = sps.poisson.cdf(cm_a, cm_xi)

            cm_mu = np.zeros(4, dtype=float)

            for cm_i in range(1, 5):

                if (cm_a - cm_i) >= 0:
                    pa = sps.poisson.cdf(cm_a, cm_xi) - sps.poisson.cdf(cm_a - cm_i, cm_xi)
                else:
                    pa = sps.poisson.cdf(cm_a, cm_xi)

                if (cm_b - cm_i - 1) >= 0:
                    pb = sps.poisson.cdf(cm_b - 1, cm_xi) - sps.poisson.cdf(cm_b - cm_i - 1, cm_xi)
                else:
                    if (cm_b - 1) >= 0:
                        pb = sps.poisson.cdf(cm_b - 1, cm_xi)
                    else:
                        pb = 0

                cm_mu[cm_i - 1] = cm_xi**cm_i * (1.0 - ((pa - pb) / d))

            cm_mom = np.zeros(5, dtype=float)
            cm_mom[0] = cm_mu[0]
            cm_mom[1] = cm_mu[1] + cm_mu[0] - cm_mu[0]**2.0
            cm_mom[2] = cm_mu[2] + (cm_mu[1] * (3.0 - (3.0 * cm_mu[0]))) + (cm_mu[0] - (3.0 * cm_mu[0]**2.0) + (2.0 * cm_mu[0]**3.0))
            cm_mom[3] = cm_mu[3] + (cm_mu[2] * (6.0 - (4.0 * cm_mu[0]))) + (cm_mu[1] * (7.0 - (12.0 * cm_mu[0]) + (6.0 * cm_mu[0]**2.0))) + cm_mu[0] - (4.0 * cm_mu[0]**2.0) + (6.0 * cm_mu[0]**3.0) - (3.0 * cm_mu[0]**4.0)
            cm_mom[4] = d

            return cm_mom

        def truncated_poisson(tp_c: int, tp_x: tarray, tp_n: int, tp_k: int) -> float:

            tp_m = np.zeros((tp_k, 5), dtype=float)

            for tp_i in range(tp_k):
                tp_m[tp_i, :] = compute_moments(tp_c, tp_x[tp_i])

            tp_m[:, 3] -= 3.0 * tp_m[:, 1]**2.0

            tp_s = np.sum(tp_m, axis=0)
            tp_z = (tp_n - tp_s[0]) / np.sqrt(tp_s[1])
            tp_g1 = tp_s[2] / tp_s[1]**1.5
            tp_g2 = tp_s[3] / tp_s[1]**2.0

            tp_e1 = tp_g1 * ((tp_z**3.0 - (3.0 * tp_z)) / 6.0)
            tp_e2 = tp_g2 * ((tp_z**4.0 - (6.0 * tp_z**2.0) + 3.0) / 24.0)
            tp_e3 = tp_g1**2.0 * ((tp_z**6.0 - (15.0 * tp_z**4.0) + (45.0 * tp_z**2.0) - 15.0) / 72.0)
            tp_poly = 1.0 + tp_e1 + tp_e2 + tp_e3

            tp_f = tp_poly * (np.exp(-tp_z**2.0 / 2.0) / (np.sqrt(2.0) * gamma(0.5)))
            tp_value = (1.0 / (sps.poisson.cdf(tp_n, tp_n) - sps.poisson.cdf(tp_n - 1, tp_n))) * np.prod(tp_m[:, 4]) * (tp_f / np.sqrt(tp_s[1]))

            return tp_value

        try:

            fitting_type = validate_enumerator(fitting_type, ['map', 'mle'])
            possible_states = validate_state_names(possible_states)
            walk = validate_states(walk, possible_states, 'walk', False)

            if fitting_type == 'map':
                if k is None:
                    k = np.ones((len(possible_states), len(possible_states)), dtype=float)
                else:
                    k = validate_hyperparameter(k, len(possible_states))
            else:
                if k is None:
                    k = False
                else:
                    k = validate_boolean(k)

            confidence_level = validate_float(confidence_level, lower_limit=(0.0, False), upper_limit=(1.0, False))

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        size = len(possible_states)
        p = np.zeros((size, size), dtype=float)

        f = np.zeros((size, size), dtype=int)

        for (i, j) in zip(walk[:-1], walk[1:]):
            f[i, j] += 1

        if fitting_type == 'map':

            for i in range(size):
                rt = np.sum(f[i, :]) + np.sum(k[i, :])

                for j in range(size):
                    ct = f[i, j] + k[i, j]

                    if rt == size:
                        p[i, j] = 1.0 / size
                    else:
                        p[i, j] = (ct - 1.0) / (rt - size)

        else:

            for (i, j) in zip(walk[:-1], walk[1:]):
                p[i, j] += 1.0

            if k:
                p += 0.001
            else:
                p[np.where(~p.any(axis=1)), :] = np.ones(size, dtype=float)

            p /= np.sum(p, axis=1, keepdims=True)

        ci_lower = np.zeros((size, size), dtype=float)
        ci_upper = np.zeros((size, size), dtype=float)

        for i in range(size):

            fi = f[i, :]
            n = np.sum(fi).item()

            c = -1
            tp = tp_previous = 0.0

            for c_current in range(1, n + 1):

                tp = truncated_poisson(c_current, fi, n, size)

                if (tp > confidence_level) and (tp_previous < confidence_level):
                    c = c_current - 1
                    break

                tp_previous = tp

            delta = (confidence_level - tp_previous) / (tp - tp_previous)
            cdn = c / n

            buffer = np.zeros((size, 5), dtype=float)
            result = np.zeros((size, 2), dtype=float)

            for j in range(size):

                obs = fi[j] / n
                buffer[j, 0] = obs
                buffer[j, 1] = max(0.0, obs - cdn)
                buffer[j, 2] = min(1.0, obs + cdn + (2.0 * (delta / n)))
                buffer[j, 3] = obs - cdn - (1.0 / n)
                buffer[j, 4] = obs + cdn + (1.0 / n)

                result[j, 0] = buffer[j, 1]
                result[j, 1] = buffer[j, 2]

            for j in range(size):

                ci_lower[i, j] = result[j, 0]
                ci_upper[i, j] = result[j, 1]

        return MarkovChain(p, possible_states), [ci_lower, ci_upper]

    @staticmethod
    def from_dictionary(d: tmc_dict_flex) -> tmc:

        """
        The method generates a Markov chain from the given dictionary, whose keys represent state pairs and whose values represent transition probabilities.

        :param d: the dictionary to transform into the transition matrix.
        :return: a Markov chain.
        :raises ValueError: if the transition matrix defined by the dictionary is not valid.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            d = validate_dictionary(d)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        states = [key[0] for key in d.keys() if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the dictionary must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the dictionary must sum to 1.')

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def from_graph(graph: tgraphs) -> tmc:

        """
        The method generates a Markov chain from the given directed graph.

        :return: a Markov chain.
        :raises ValueError: if the transition matrix defined by the directed graph is not valid.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            graph = validate_graph(graph)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        states = list(graph.nodes)
        size = len(states)

        p = np.zeros((size, size), dtype=float)

        for state_from, weights in graph.adjacency():
            i = states.index(state_from)
            for state_to, data in weights.items():
                j = states.index(state_to)
                w = data['weight']
                p[i, j] = w

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def from_file(file_path: str) -> tmc:

        """
        The method reads a Markov chain from the given file.

        | Only csv, json, xml and plain text files are supported; data format is inferred from the file extension.
        |
        | In *csv* files, the header must contain the state names and every row must represent a row of the transition matrix.
        | The following format settings are required:
        | delimiter: *,*
        | quoting: *minimal*
        | quote character: *"*
        |
        | In *json* files, data must be structured as an array of objects with the following properties:
        | *state_from* (string)
        | *state_to* (string)
        | *probability* (float or int)
        |
        | In *text* files, every line of the file must have the following format:
        | *<state_from> <state_to> <probability>*
        |
        | In *xml* files, the root element must be called *MarkovChain* and child elements must be called *Transition*.
        | Every child element must have the following attributes:
        | *state_from* (string)
        | *state_to* (string)
        | *probability* (float or int)

        :param file_path: the location of the file that defines the Markov chain.
        :return: a Markov chain.
        :raises FileNotFoundError: if the file does not exist.
        :raises OSError: if the file cannot be read or is empty.
        :raises ValidationError: if any input argument is not compliant.
        :raises ValueError: if the file contains invalid data.
        """

        try:

            file_path = validate_string(file_path)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        file_extension = get_file_extension(file_path)

        if file_extension not in ['.csv', '.json', '.xml', '.txt']:  # pragma: no cover
            raise ValidationError('Only csv, json, xml and plain text files are supported.')

        if file_extension == '.csv':
            d = read_csv(file_path)
        elif file_extension == '.json':
            d = read_json(file_path)
        elif file_extension == '.txt':
            d = read_txt(file_path)
        else:
            d = read_xml(file_path)

        states = [key[0] for key in d.keys() if key[0] == key[1]]
        size = len(states)

        if size < 2:  # pragma: no cover
            raise ValueError('The size of the transition matrix defined by the file must be greater than or equal to 2.')

        p = np.zeros((size, size), dtype=float)

        for it, ip in d.items():
            p[states.index(it[0]), states.index(it[1])] = ip

        if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):  # pragma: no cover
            raise ValueError('The rows of the transition matrix defined by the file must sum to 1.')

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def from_matrix(m: tnumeric, states: olist_str = None) -> tmc:

        """
        The method generates a Markov chain with the given state names, whose transition matrix is obtained through the normalization of the given matrix.

        :param m: the matrix to transform into the transition matrix.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            m = validate_matrix(m)

            if states is None:
                states = [str(i) for i in range(1, m.shape[0] + 1)]
            else:  # pragma: no cover
                states = validate_state_names(states, m.shape[0])

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        m = np.interp(m, (np.min(m), np.max(m)), (0.0, 1.0))
        m /= np.sum(m, axis=1, keepdims=True)
        m = np.nan_to_num(m, 1.0 / m.shape[0])

        return MarkovChain(m, states)

    @staticmethod
    def gamblers_ruin(size: int, w: float, states: olist_str = None) -> tmc:

        """
        The method generates a gambler's ruin Markov chain of given size and win probability.

        :param size: the size of the Markov chain.
        :param w: the win probability.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = validate_integer(size, lower_limit=(3, False))
            w = validate_float(w, lower_limit=(0.0, True), upper_limit=(1.0, True))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:  # pragma: no cover
                states = validate_state_names(states, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, _ = gamblers_ruin(size, w)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def identity(size: int, states: olist_str = None) -> tmc:

        """
        The method generates a Markov chain of given size based on an identity transition matrix.

        :param size: the size of the Markov chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            size = validate_integer(size, lower_limit=(2, False))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:  # pragma: no cover
                states = validate_state_names(states, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p = np.eye(size, dtype=float)
        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def random(size: int, states: olist_str = None, zeros: int = 0, mask: onumeric = None, seed: oint = None) -> tmc:

        """
        The method generates a Markov chain of given size with random transition probabilities.

        :param size: the size of the Markov chain.
        :param states: the name of each state (if omitted, an increasing sequence of integers starting at 1).
        :param zeros: the number of zero-valued transition probabilities (by default, 0).
        :param mask: a matrix representing the locations and values of fixed transition probabilities (random transition probabilities are represented by NaN values).
        :param seed: a seed to be used as RNG initializer for reproducibility purposes.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            rng = create_rng(seed)
            size = validate_integer(size, lower_limit=(2, False))

            if states is None:
                states = [str(i) for i in range(1, size + 1)]
            else:  # pragma: no cover
                states = validate_state_names(states, size)

            zeros = validate_integer(zeros, lower_limit=(0, False))

            if mask is None:
                mask = np.full((size, size), np.nan, dtype=float)
            else:
                mask = validate_mask(mask, size)

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, error_message = random(rng, size, zeros, mask)

        if error_message is not None:  # pragma: no cover
            raise ValidationError(error_message)

        mc = MarkovChain(p, states)

        return mc

    @staticmethod
    def urn_model(n: int, model: str) -> tmc:

        """
        The method generates a Markov chain of size *2n + 1* based on the specified urn model.

        :param n: the number of elements in each urn.
        :param model:
         - *bernoulli-laplace* for the Bernoulli-Laplace urn model;
         - *ehrenfest* for the Ehrenfest urn model.
        :return: a Markov chain.
        :raises ValidationError: if any input argument is not compliant.
        """

        try:

            n = validate_integer(n, lower_limit=(1, False))
            model = validate_enumerator(model, ['bernoulli-laplace', 'ehrenfest'])

        except Exception as e:  # pragma: no cover
            raise generate_validation_error(e, trace()) from None

        p, states, _ = urn_model(n, model)
        mc = MarkovChain(p, states)

        return mc
