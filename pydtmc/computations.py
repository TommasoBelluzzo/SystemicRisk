# -*- coding: utf-8 -*-

__all__ = [
    'eigenvalues_sorted',
    'gth_solve',
    'rdl_decomposition',
    'slem'
]


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.linalg as npl

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def eigenvalues_sorted(m: tarray) -> tarray:

    ev = npl.eigvals(m)
    ev = np.sort(np.abs(ev))

    return ev


def gth_solve(p: tarray) -> tarray:

    a = np.copy(p)
    n = a.shape[0]

    for i in range(n - 1):

        scale = np.sum(a[i, i + 1:n])

        if scale <= 0.0:  # pragma: no cover
            n = i + 1
            break

        a[i + 1:n, i] /= scale
        a[i + 1:n, i + 1:n] += np.dot(a[i + 1:n, i:i + 1], a[i:i + 1, i + 1:n])

    x = np.zeros(n, dtype=float)
    x[n - 1] = 1.0

    for i in range(n - 2, -1, -1):
        x[i] = np.dot(x[i + 1:n], a[i + 1:n, i])

    x /= np.sum(x)

    return x


def rdl_decomposition(p: tarray) -> trdl:

    values, vectors = npl.eig(p)

    indices = np.argsort(np.abs(values))[::-1]
    values = values[indices]
    vectors = vectors[:, indices]

    r = np.copy(vectors)
    d = np.diag(values)
    l = npl.solve(np.transpose(r), np.eye(p.shape[0], dtype=float))

    k = np.sum(l[:, 0])

    if not np.isclose(k, 0.0):
        r[:, 0] *= k
        l[:, 0] /= k

    r = np.real(r)
    d = np.real(d)
    l = np.transpose(np.real(l))

    return r, d, l


def slem(m: tarray) -> ofloat:

    ev = eigenvalues_sorted(m)
    indices = np.isclose(ev, 1.0)

    value = ev[~indices][-1]

    if np.isclose(value, 0.0):
        return None

    return value
