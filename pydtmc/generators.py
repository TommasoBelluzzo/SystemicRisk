# -*- coding: utf-8 -*-

__all__ = [
    'approximation',
    'birth_death',
    'bounded',
    'canonical',
    'closest_reversible',
    'gamblers_ruin',
    'lazy',
    'lump',
    'random',
    'sub',
    'urn_model'
]


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.linalg as npl
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.stats as sps

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def approximation(size: int, approximation_type: str, alpha: float, sigma: float, rho: float, k: ofloat) -> tgenres_ext:

    def adda_cooper_integrand(aci_x, aci_sigma_z, aci_sigma, aci_rho, aci_alpha, z_j, z_jp1):

        t1 = np.exp((-1.0 * (aci_x - aci_alpha) ** 2.0) / (2.0 * aci_sigma_z ** 2.0))
        t2 = sps.norm.cdf((z_jp1 - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)
        t3 = sps.norm.cdf((z_j - (aci_alpha * (1.0 - aci_rho)) - (aci_rho * aci_x)) / aci_sigma)

        return t1 * (t2 - t3)

    def rouwenhorst_matrix(rm_size: int, rm_z: float) -> tarray:

        if rm_size == 2:
            p = np.array([[rm_z, 1 - rm_z], [1 - rm_z, rm_z]])
        else:

            t1 = np.zeros((rm_size, rm_size))
            t2 = np.zeros((rm_size, rm_size))
            t3 = np.zeros((rm_size, rm_size))
            t4 = np.zeros((rm_size, rm_size))

            theta_inner = rouwenhorst_matrix(rm_size - 1, rm_z)

            t1[:rm_size - 1, :rm_size - 1] = rm_z * theta_inner
            t2[:rm_size - 1, 1:] = (1.0 - rm_z) * theta_inner
            t3[1:, :-1] = (1.0 - rm_z) * theta_inner
            t4[1:, 1:] = rm_z * theta_inner

            p = t1 + t2 + t3 + t4
            p[1:rm_size - 1, :] /= 2.0

        return p

    if approximation_type == 'adda-cooper':

        z_sigma = sigma / (1.0 - rho ** 2.00) ** 0.5
        z = (z_sigma * sps.norm.ppf(np.arange(size + 1) / size)) + alpha

        p = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                iq = spi.quad(adda_cooper_integrand, z[i], z[i + 1], args=(z_sigma, sigma, rho, alpha, z[j], z[j + 1]))
                p[i, j] = (size / np.sqrt(2.0 * np.pi * z_sigma ** 2.0)) * iq[0]

    elif approximation_type == 'rouwenhorst':

        z = (1.0 + rho) / 2.0
        p = rouwenhorst_matrix(size, z)

    elif approximation_type == 'tauchen-hussey':

        nodes = np.zeros(size, dtype=float)
        weights = np.zeros(size, dtype=float)

        pp = 0.0
        z = 0.0

        for i in range(int(np.fix((size + 1) / 2))):

            if i == 0:
                z = np.sqrt((2.0 * size) + 1.0) - (1.85575 * ((2.0 * size) + 1.0) ** -0.16393)
            elif i == 1:
                z = z - ((1.14 * size ** 0.426) / z)
            elif i == 2:
                z = (1.86 * z) + (0.86 * nodes[0])
            elif i == 3:
                z = (1.91 * z) + (0.91 * nodes[1])
            else:
                z = (2.0 * z) + nodes[i - 2]

            iterations = 0

            while iterations < 100:

                iterations += 1

                p1 = 1.0 / np.pi ** 0.25
                p2 = 0.0

                for j in range(1, size + 1):
                    p3 = p2
                    p2 = p1
                    p1 = (z * np.sqrt(2.0 / j) * p2) - (np.sqrt((j - 1.0) / j) * p3)

                pp = np.sqrt(2.0 * size) * p2

                z1 = z
                z = z1 - p1 / pp

                if np.abs(z - z1) < 1e-14:
                    break

            if iterations == 100:
                return None, None, 'The gaussian quadrature failed to converge.'

            nodes[i] = -z
            nodes[size - i - 1] = z

            weights[i] = 2.0 / pp ** 2.0
            weights[size - i - 1] = weights[i]

        nodes = (nodes * np.sqrt(2.0) * np.sqrt(2.0 * k ** 2.0)) + alpha
        weights = weights / np.sqrt(np.pi) ** 2.0

        p = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                prime = ((1.0 - rho) * alpha) + (rho * nodes[i])
                p[i, j] = (weights[j] * sps.norm.pdf(nodes[j], prime, sigma) / sps.norm.pdf(nodes[j], alpha, k))

        for i in range(size):
            p[i, :] /= np.sum(p[i, :])

    else:

        if np.array_equal(rho, 1.0):
            rho = 0.999999999999999

        y_std = np.sqrt(sigma ** 2.0 / (1.0 - rho**2.0))

        x_max = y_std * k
        x_min = -x_max
        x = np.linspace(x_min, x_max, size)

        step = 0.5 * ((x_max - x_min) / (size - 1))
        p = np.zeros((size, size), dtype=float)

        for i in range(size):
            p[i, 0] = sps.norm.cdf((x[0] - (rho * x[i]) + step) / sigma)
            p[i, size - 1] = 1.0 - sps.norm.cdf((x[size - 1] - (rho * x[i]) - step) / sigma)

            for j in range(1, size - 1):
                z = x[j] - (rho * x[i])
                p[i, j] = sps.norm.cdf((z + step) / sigma) - sps.norm.cdf((z - step) / sigma)

    states = ['A' + str(i) for i in range(1, p.shape[0] + 1)]

    return p, states, None


def birth_death(p: tarray, q: tarray) -> tgenres:

    r = 1.0 - q - p

    p = np.diag(r, k=0) + np.diag(p[0:-1], k=1) + np.diag(q[1:], k=-1)
    p[np.isclose(p, 0.0)] = 0.0
    p /= np.sum(p, axis=1, keepdims=True)

    return p, None


def bounded(p: tarray, boundary_condition: tbcond) -> tgenres:

    size = p.shape[0]

    first = np.zeros(size, dtype=float)
    last = np.zeros(size, dtype=float)

    if isinstance(boundary_condition, float):

        first[0] = 1.0 - boundary_condition
        first[1] = boundary_condition
        last[-1] = boundary_condition
        last[-2] = 1.0 - boundary_condition

    else:

        if boundary_condition == 'absorbing':
            first[0] = 1.0
            last[-1] = 1.0
        else:
            first[1] = 1.0
            last[-2] = 1.0

    p_adjusted = np.copy(p)
    p_adjusted[0] = first
    p_adjusted[-1] = last

    return p_adjusted, None


def canonical(p: tarray, recurrent_indices: tlist_int, transient_indices: tlist_int) -> tgenres:

    p = np.copy(p)

    if len(recurrent_indices) == 0 or len(transient_indices) == 0:
        return p, None

    is_canonical = max(transient_indices) < min(recurrent_indices)

    if is_canonical:
        return p, None

    indices = transient_indices + recurrent_indices

    p = p[np.ix_(indices, indices)]

    return p, None


def closest_reversible(p: tarray, distribution: tnumeric, weighted: bool) -> tgenres:

    def jacobian(xj: tarray, hj: tarray, fj: tarray):
        return np.dot(np.transpose(xj), hj) + fj

    def objective(xo: tarray, ho: tarray, fo: tarray):
        return (0.5 * npl.multi_dot([np.transpose(xo), ho, xo])) + np.dot(np.transpose(fo), xo)

    size = p.shape[0]

    zeros = len(distribution) - np.count_nonzero(distribution)
    m = int((((size - 1) * size) / 2) + (((zeros - 1) * zeros) / 2) + 1)

    basis_vectors = []

    for r in range(size - 1):
        for s in range(r + 1, size):

            if distribution[r] == 0.0 and distribution[s] == 0.0:

                bv = np.eye(size, dtype=float)
                bv[r, r] = 0.0
                bv[r, s] = 1.0
                basis_vectors.append(bv)

                bv = np.eye(size, dtype=float)
                bv[r, r] = 1.0
                bv[r, s] = 0.0
                bv[s, s] = 0.0
                bv[s, r] = 1.0
                basis_vectors.append(bv)

            else:

                bv = np.eye(size, dtype=float)
                bv[r, r] = 1.0 - distribution[s]
                bv[r, s] = distribution[s]
                bv[s, s] = 1.0 - distribution[r]
                bv[s, r] = distribution[r]
                basis_vectors.append(bv)

    basis_vectors.append(np.eye(size, dtype=float))

    h = np.zeros((m, m), dtype=float)
    f = np.zeros(m, dtype=float)

    if weighted:

        d = np.diag(distribution)
        di = npl.inv(d)

        for i in range(m):

            bv_i = basis_vectors[i]
            z = npl.multi_dot([d, bv_i, di])

            f[i] = -2.0 * np.trace(np.dot(z, np.transpose(p)))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * np.trace(np.dot(np.transpose(z), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    else:

        for i in range(m):

            bv_i = basis_vectors[i]
            f[i] = -2.0 * np.trace(np.dot(np.transpose(bv_i), p))

            for j in range(m):
                bv_j = basis_vectors[j]

                tau = 2.0 * np.trace(np.dot(np.transpose(bv_i), bv_j))
                h[i, j] = tau
                h[j, i] = tau

    a = np.zeros((m + size - 1, m), dtype=float)
    np.fill_diagonal(a, -1.0)
    a[m - 1, m - 1] = 0.0

    for i in range(size):

        k = 0

        for r in range(size - 1):
            for s in range(r + 1, size):

                if distribution[s] == 0.0 and distribution[r] == 0.0:

                    if r != i:
                        a[m + i - 1, k] = -1.0
                    else:
                        a[m + i - 1, k] = 0.0

                    k += 1

                    if s != i:
                        a[m + i - 1, k] = -1.0
                    else:
                        a[m + i - 1, k] = 0.0

                elif s == i:
                    a[m + i - 1, k] = -1.0 + distribution[r]
                elif r == i:
                    a[m + i - 1, k] = -1.0 + distribution[s]
                else:
                    a[m + i - 1, k] = -1.0

                k += 1

        a[m + i - 1, m - 1] = -1.0

    b = np.zeros(m + size - 1, dtype=float)
    x0 = np.zeros(m, dtype=float)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: b - np.dot(a, x), 'jac': lambda x: -a}
    )

    # noinspection PyTypeChecker
    solution = spo.minimize(objective, x0, jac=jacobian, args=(h, f), constraints=constraints, method='SLSQP', options={'disp': False})

    if not solution['success']:  # pragma: no cover
        return None, 'The closest reversible could not be computed.'

    p = np.zeros((size, size), dtype=float)
    solution = solution['x']

    for i in range(m):
        p += solution[i] * basis_vectors[i]

    p /= np.sum(p, axis=1, keepdims=True)

    return p, None


def gamblers_ruin(size: int, w: float) -> tgenres:

    p = np.zeros((size, size), dtype=float)
    p[0, 0] = 1.0
    p[-1, -1] = 1.0

    for i in range(1, size - 1):
        p[i, i - 1] = 1.0 - w
        p[i, i + 1] = w

    return p, None


def lazy(p: tarray, inertial_weights: tarray) -> tgenres:

    size = p.shape[0]

    p1 = (1.0 - inertial_weights)[:, np.newaxis] * p
    p2 = np.eye(size, dtype=float) * inertial_weights
    p = p1 + p2

    return p, None


def lump(p: tarray, states: tlist_str, partitions: tlists_int) -> tgenres_ext:

    size = p.shape[0]

    r = np.zeros((size, len(partitions)), dtype=float)

    for index, partition in enumerate(partitions):
        for state in partition:
            r[state, index] = 1.0

    # noinspection PyBroadException
    try:
        k = np.dot(np.linalg.inv(np.dot(np.transpose(r), r)), np.transpose(r))
    except Exception:  # pragma: no cover
        return None, None, 'The Markov chain is not strongly lumpable with respect to the given partitions.'

    left = np.dot(np.dot(np.dot(r, k), p), r)
    right = np.dot(p, r)
    is_lumpable = np.array_equal(left, right)

    if not is_lumpable:  # pragma: no cover
        return None, None, 'The Markov chain is not strongly lumpable with respect to the given partitions.'

    p_lump = np.dot(np.dot(k, p), r)

    # noinspection PyTypeChecker
    state_names = [','.join(list(map(states.__getitem__, partition))) for partition in partitions]

    return p_lump, state_names, None


def random(rng: trand, size: int, zeros: int, mask: tarray) -> tgenres:

    full_rows = np.isclose(np.nansum(mask, axis=1, dtype=float), 1.0)

    mask_full = np.transpose(np.array([full_rows, ] * size))
    mask[np.isnan(mask) & mask_full] = 0.0

    mask_unassigned = np.isnan(mask)
    zeros_required = (np.sum(mask_unassigned) - np.sum(~full_rows)).item()

    if zeros > zeros_required:  # pragma: no cover
        return None, f'The number of zero-valued transition probabilities exceeds the maximum threshold of {zeros_required:d}.'

    n = np.arange(size)

    for i in n:
        if not full_rows[i]:
            row = mask_unassigned[i, :]
            columns = np.flatnonzero(row)
            j = columns[rng.randint(0, np.sum(row).item())]
            mask[i, j] = np.inf

    mask_unassigned = np.isnan(mask)
    indices_unassigned = np.flatnonzero(mask_unassigned)

    r = rng.permutation(zeros_required)
    indices_zero = indices_unassigned[r[0:zeros]]
    indices_rows, indices_columns = np.unravel_index(indices_zero, (size, size))

    mask[indices_rows, indices_columns] = 0.0
    mask[np.isinf(mask)] = np.nan

    p = np.copy(mask)
    p_unassigned = np.isnan(mask)
    p[p_unassigned] = np.ravel(rng.rand(1, np.sum(p_unassigned, dtype=int).item()))

    for i in n:

        assigned_columns = np.isnan(mask[i, :])
        s = np.sum(p[i, assigned_columns])

        if s > 0.0:
            si = np.sum(p[i, ~assigned_columns])
            p[i, assigned_columns] = p[i, assigned_columns] * ((1.0 - si) / s)

    return p, None


def sub(p: tarray, states: tlist_str, adjacency_matrix: tarray, sub_states: tlist_int) -> tgenres_ext:

    size = p.shape[0]

    closure = np.copy(adjacency_matrix)

    for i in range(size):
        for j in range(size):
            for x in range(size):
                closure[j, x] = closure[j, x] or (closure[j, i] and closure[i, x])

    for state in sub_states:
        for sc in np.ravel([np.where(closure[state, :] == 1.0)]):
            if sc not in sub_states:
                sub_states.append(sc)

    sub_states = sorted(sub_states)

    p = np.copy(p)
    p = p[np.ix_(sub_states, sub_states)]

    if p.size == 1:  # pragma: no cover
        return None, None, 'The subchain is not a valid Markov chain.'

    state_names = [*map(states.__getitem__, sub_states)]

    return p, state_names, None


def urn_model(n: int, model: str) -> tgenres_ext:

    dn = n * 2
    size = dn + 1

    p = np.zeros((size, size), dtype=float)
    p_row = np.repeat(0.0, size)

    if model == 'bernoulli-laplace':

        for i in range(size):

            r = np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = (i / dn) ** 2.0
                r[i] = 2.0 * (i / dn) * (1.0 - (i / dn))
                r[i + 1] = (1.0 - (i / dn)) ** 2.0

            p[i, :] = r

    else:

        for i in range(size):

            r = np.copy(p_row)

            if i == 0:
                r[1] = 1.0
            elif i == dn:
                r[-2] = 1.0
            else:
                r[i - 1] = i / dn
                r[i + 1] = 1.0 - (i / dn)

            p[i, :] = r

    state_names = [f'U{i}' for i in range(1, (n * 2) + 2)]

    return p, state_names, None
