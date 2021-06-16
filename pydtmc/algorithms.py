# -*- coding: utf-8 -*-

__all__ = [
    'calculate_periods',
    'find_cyclic_classes',
    'find_lumping_partitions'
]


###########
# IMPORTS #
###########

# Full

import networkx as nx
import numpy as np

# Partial

from itertools import (
    chain
)

from math import (
    gcd
)

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def calculate_period(graph: tgraph) -> int:

    g = 0

    for scc in nx.strongly_connected_components(graph):

        scc = list(scc)

        levels = dict((scc, None) for scc in scc)
        vertices = levels

        x = scc[0]
        levels[x] = 0

        current_level = [x]
        previous_level = 1

        while current_level:

            next_level = []

            for u in current_level:
                for v in graph[u]:

                    if v not in vertices:  # pragma: no cover
                        continue

                    level = levels[v]

                    if level is not None:

                        g = gcd(g, previous_level - level)

                        if g == 1:
                            return 1

                    else:

                        next_level.append(v)
                        levels[v] = previous_level

            current_level = next_level
            previous_level += 1

    return g


def calculate_periods(graph: tgraph) -> tlist_int:

    sccs = list(nx.strongly_connected_components(graph))

    classes = [sorted([c for c in scc]) for scc in sccs]
    indices = sorted(classes, key=lambda x: (-len(x), x[0]))

    periods = [0] * len(indices)

    for scc in sccs:

        scc_reachable = scc.copy()

        for c in scc_reachable:
            spl = nx.shortest_path_length(graph, c).keys()
            scc_reachable = scc_reachable.union(spl)

        index = indices.index(sorted(list(scc)))

        if (scc_reachable - scc) == set():
            periods[index] = calculate_period(graph.subgraph(scc))
        else:
            periods[index] = 1

    return periods


def find_cyclic_classes(p: tarray) -> tarray:

    size = p.shape[0]

    v = np.zeros(size, dtype=int)
    v[0] = 1

    w = np.array([], dtype=int)
    t = np.array([0], dtype=int)

    d = 0
    m = 1

    while (m > 0) and (d != 1):

        i = t[0]
        j = 0

        t = np.delete(t, 0)
        w = np.append(w, i)

        while j < size:

            if p[i, j] > 0.0:
                r = np.append(w, t)
                k = np.sum(r == j)

                if k > 0:
                    b = v[i] - v[j] + 1
                    d = gcd(d, b)
                else:
                    t = np.append(t, j)
                    v[j] = v[i] + 1

            j += 1

        m = t.size

    v = np.remainder(v, d)

    indices = list()

    for u in np.unique(v):
        indices.append(list(chain.from_iterable(np.argwhere(v == u))))

    return indices


def find_lumping_partitions(p: tarray) -> tparts:

    size = p.shape[0]

    if size == 2:
        return []

    k = size - 1
    indices = list(range(size))

    possible_partitions = []

    for i in range(2**k):

        partition = []
        subset = []

        for position in range(size):

            subset.append(indices[position])

            if ((1 << position) & i) or position == k:
                partition.append(subset)
                subset = []

        partition_length = len(partition)

        if 2 <= partition_length < size:
            possible_partitions.append(partition)

    partitions = []

    for partition in possible_partitions:

        r = np.zeros((size, len(partition)), dtype=float)

        for i, lumping in enumerate(partition):
            for state in lumping:
                r[state, i] = 1.0

        # noinspection PyBroadException
        try:
            k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
        except Exception:  # pragma: no cover
            continue

        left = np.dot(np.dot(np.dot(r, k), p), r)
        right = np.dot(p, r)
        is_lumpable = np.array_equal(left, right)

        if is_lumpable:
            partitions.append(partition)

    return partitions
