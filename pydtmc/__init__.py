# -*- coding: utf-8 -*-

__title__ = 'PyDTMC'
__version__ = '5.5.0'
__author__ = 'Tommaso Belluzzo'

__all__ = [
    'ValidationError',
    'MarkovChain',
    'plot_eigenvalues', 'plot_graph', 'plot_redistributions', 'plot_walk'
]

from pydtmc.exceptions import (
    ValidationError
)

from pydtmc.markov_chain import (
    MarkovChain
)

from pydtmc.plotting import (
    plot_eigenvalues,
    plot_graph,
    plot_redistributions,
    plot_walk
)


import numpy as np
from scipy.linalg import eig, eigh, lu_factor, lu_solve, solve

def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.
    Parameters
    ----------
    A : (N, N) ndarray
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerace parameter for termination of iteration
    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu
    """
    T = A - mu * np.eye(A.shape[0])
    """LU-factor of T"""
    lupiv = lu_factor(T)
    """Starting iterate with ||y_0||=1"""
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    """Local variables for inverse iteration"""
    y = 1.0 * y0
    r = 1.0 * r0
    for i in range(maxiter):
        x = lu_solve(lupiv, y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" % (maxiter, r)
    raise RuntimeError(msg)


def stationary_distribution_from_backward_iteration(P, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.
    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.
    Returns
    -------
    pi : (M,) ndarray
        Stationary vector
    """
    A = np.transpose(P)
    mu = 1.0 - eps
    x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)
    pi = y / y.sum()
    return pi


def stationary_distribution_from_eigenvector(T):
    r"""Compute stationary distribution of stochastic matrix T.
    The stationary distribution is the left eigenvector corresponding to the
    non-degenerate eigenvalue :math: `\lambda=1`.
    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.
    """
    val, L = eig(T, left=True, right=False)

    """ Sorted eigenvalues and left and right eigenvectors. """
    perm = np.argsort(val)[::-1]

    val = val[perm]
    L = L[:, perm]
    """ Make sure that stationary distribution is non-negative and l1-normalized """
    nu = np.abs(L[:, 0])
    mu = nu / np.sum(nu)
    return mu


def stationary_distribution(T):
    r"""Compute stationary distribution of stochastic matrix T.
    Chooses the fastest applicable algorithm automatically
    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.
    """
    fallback = False
    try:
        mu = stationary_distribution_from_backward_iteration(T)
        if np.any(mu < 0):  # numerical problem, fall back to more robust algorithm.
            fallback=True
    except RuntimeError:
        fallback = True

    if fallback:
        mu = stationary_distribution_from_eigenvector(T)
        if np.any(mu < 0):  # still? Then set to 0 and renormalize
            mu = np.maximum(mu, 0.0)
            mu /= mu.sum()

    return mu

def is_reversible(T, mu=None, tol=1e-10):
    r"""
    checks whether T is reversible in terms of given stationary distribution.
    If no distribution is given, it will be calculated out of T.
    It performs following check:
    :math:`\pi_i P_{ij} = \pi_j P_{ji}`
    Parameters
    ----------
    T : numpy.ndarray matrix
        Transition matrix
    mu : numpy.ndarray vector
        stationary distribution
    tol : float
        tolerance to check with
    Returns
    -------
    Truth value : bool
        True, if T is a reversible transitition matrix
        False, otherwise
    """

    if mu is None:
        mu = stationary_distribution(T)
    X = mu[:, np.newaxis] * T
    return np.allclose(X, np.transpose(X),  atol=tol)


def rdl_decomposition(T, k=None, reversible=False, norm='standard', mu=None):
    r"""Compute the decomposition into left and right eigenvectors.
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible', 'auto'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
        auto: will be reversible if T is reversible, otherwise standard
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (d,) ndarray, optional
        Stationary distribution of T
    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``
    Notes
    -----
    If reversible=True the the eigenvalues and eigenvectors of the
    similar symmetric matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be
    used to compute the eigenvalues and eigenvectors of T.
    The precomputed stationary distribution will only be used if
    reversible=True.
    """
    # auto-set norm
    if norm == 'auto':
        if is_reversible(T):
            norm = 'reversible'
        else:
            norm = 'standard'

    if reversible:
        R, D, L = rdl_decomposition_rev(T, norm=norm, mu=mu)
    else:
        R, D, L = rdl_decomposition_nrev(T, norm=norm)

    if reversible or norm == 'reversible':
        D = D.real

    if k is None:
        return R, D, L
    else:
        return R[:, 0:k], D[0:k, 0:k], L[0:k, :]


def rdl_decomposition_nrev(T, norm='standard'):
    r"""Decomposition into left and right eigenvectors.
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1
        reversible: R and L are related via L=L[:,0]*R
    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``
    """
    d = T.shape[0]
    w, R = eig(T)

    """Sort by decreasing magnitude of eigenvalue"""
    ind = np.argsort(np.abs(w))[::-1]
    w = w[ind]
    R = R[:, ind]

    """Diagonal matrix containing eigenvalues"""
    D = np.diag(w)

    # Standard norm: Euclidean norm is 1 for r and LR = I.
    if norm == 'standard':
        L = solve(np.transpose(R), np.eye(d))

        """l1- normalization of L[:, 0]"""
        R[:, 0] = R[:, 0] * np.sum(L[:, 0])
        L[:, 0] = L[:, 0] / np.sum(L[:, 0])

        return R, D, np.transpose(L)

    # Reversible norm:
    elif norm == 'reversible':
        b = np.zeros(d)
        b[0] = 1.0

        A = np.transpose(R)
        nu = solve(A, b)
        mu = nu / np.sum(nu)

        """Ensure that R[:,0] is positive"""
        R[:, 0] = R[:, 0] / np.sign(R[0, 0])

        """Use mu to connect L and R"""
        L = mu[:, np.newaxis] * R

        """Compute overlap"""
        s = np.diag(np.dot(np.transpose(L), R))

        """Renormalize left-and right eigenvectors to ensure L'R=Id"""
        R = R / np.sqrt(s[np.newaxis, :])
        L = L / np.sqrt(s[np.newaxis, :])

        return R, D, np.transpose(L)

    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")


def rdl_decomposition_rev(T, norm='reversible', mu=None):
    r"""Decomposition into left and right eigenvectors for reversible
    transition matrices.
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
    mu : (M,) ndarray, optional
        Stationary distribution of T
    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the
        column R[:,i] is the right eigenvector corresponding to the eigenvalue
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``
    Notes
    -----
    The eigenvalues and eigenvectors of the similar symmetric matrix
    `\sqrt(\mu_i / \mu_j) p_{ij}` will be used to compute the
    eigenvalues and eigenvectors of T.
    The stationay distribution will be computed if no precomputed stationary
    distribution is given.
    """
    if mu is None:
        mu = stationary_distribution(T)
    """ symmetrize T """
    smu = np.sqrt(mu)
    S = smu[:,None] * T / smu
    val, eigvec = eigh(S)
    """Sort eigenvalues and eigenvectors"""
    perm = np.argsort(np.abs(val))[::-1]
    val = val[perm]
    eigvec = eigvec[:, perm]

    """Diagonal matrix of eigenvalues"""
    D = np.diag(val)

    """Right and left eigenvectors"""
    R = eigvec / smu[:, np.newaxis]
    L = eigvec * smu[:, np.newaxis]

    """Ensure that R[:,0] is positive and unity"""
    tmp = R[0, 0]
    R[:, 0] = R[:, 0] / tmp

    """Ensure that L[:, 0] is probability vector"""
    L[:, 0] = L[:, 0] *  tmp

    if norm == 'reversible':
        return R, D, L.T
    elif norm == 'standard':
        """Standard l2-norm of right eigenvectors"""
        w = np.diag(np.dot(R.T, R))
        sw = np.sqrt(w)
        """Don't change normalization of eigenvectors for dominant eigenvalue"""
        sw[0] = 1.0

        R = R / sw[np.newaxis, :]
        L = L * sw[np.newaxis, :]
        return R, D, L.T
    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")


def time_correlation_by_diagonalization(P, pi, obs1, obs2=None, time=1, rdl=None):
    """
    calculates time correlation. Raises P to power 'times' by diagonalization.
    If rdl tuple (R, D, L) is given, it will be used for
    further calculation.
    """
    if rdl is None:
        raise ValueError("no rdl decomposition")
    R, D, L = rdl

    d_times = np.diag(D) ** time
    diag_inds = np.diag_indices_from(D)
    D_time = np.zeros(D.shape, dtype=d_times.dtype)
    D_time[diag_inds] = d_times
    P_time = np.dot(np.dot(R, D_time), L)

    # multiply element-wise obs1 and pi. this is obs1' diag(pi)
    l = np.multiply(obs1, pi)
    m = np.dot(P_time, obs2)
    result = np.dot(l, m)
    return result


def time_correlation_direct_by_mtx_vec_prod(P, mu, obs1, obs2=None, time=1, start_values=None, return_P_k_obs=False):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    The time-correlation at time=k is computed by the matrix-vector expression:
    cor(k) = obs1' diag(pi) P^k obs2
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    mu : ndarray, shape=(n)
        stationary distribution vector.
    time : int
        time point at which the (auto)correlation will be evaluated.
    start_values : (time, ndarray <P, <P, obs2>>_t)
        start iteration of calculation of matrix power product, with this values.
        only useful when calling this function out of a loop over times.
    return_P_k_obs : bool
        if True, the dot product <P^time, obs2> will be returned for further
        calculations.
    Returns
    -------
    cor(k) : float
           correlation between observations
    """
    # input checks
    if not (type(time) == int):
        if not (type(time) == np.int64):
            raise TypeError("given time (%s) is not an integer, but has type: %s"
                            % (str(time), type(time)))
    if obs1.shape[0] != P.shape[0]:
        raise ValueError("observable shape not compatible with given matrix")
    if obs2 is None:
        obs2 = obs1
    # multiply element-wise obs1 and pi. this is obs1' diag(pi)
    l = np.multiply(obs1, mu)
    # raise transition matrix to power of time by substituting dot product
    # <Pk, obs2> with something like <P, <P, obs2>>.
    # This saves a lot of matrix matrix multiplications.
    if start_values:  # begin with a previous calculated val
        P_i_obs = start_values[1]
        # calculate difference properly!
        time_prev = start_values[0]
        t_diff = time - time_prev
        r = range(t_diff)
    else:
        if time >= 2:
            P_i_obs = np.dot(P, np.dot(P, obs2))  # vector <P, <P, obs2> := P^2 * obs
            r = range(time - 2)
        elif time == 1:
            P_i_obs = np.dot(P, obs2)  # P^1 = P*obs
            r = range(0)
        elif time == 0:  # P^0 = I => I*obs2 = obs2
            P_i_obs = obs2
            r = range(0)

    for k in r:  # since we already substituted started with 0
        P_i_obs = np.dot(P, P_i_obs)
    corr = np.dot(l, P_i_obs)
    if return_P_k_obs:
        return corr, (time, P_i_obs)
    else:
        return corr


def time_correlations_direct(P, pi, obs1, obs2=None, times=[1]):
    r"""Compute time-correlations of obs1, or time-cross-correlation with obs2.
    The time-correlation at time=k is computed by the matrix-vector expression:
    cor(k) = obs1' diag(pi) P^k obs2
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    pi : ndarray, shape=(n)
        stationary distribution vector. Will be computed if not given
    times : array-like, shape(n_t)
        Vector of time points at which the (auto)correlation will be evaluated
    Returns
    -------
    """
    n_t = len(times)
    times = np.sort(times)  # sort it to use caching of previously computed correlations
    f = np.zeros(n_t)

    # maximum time > number of rows?
    if times[-1] > P.shape[0]:
        use_diagonalization = True
        R, D, L = rdl_decomposition(P)
        # discard imaginary part, if all elements i=0
        if not np.any(np.iscomplex(R)):
            R = np.real(R)
        if not np.any(np.iscomplex(D)):
            D = np.real(D)
        if not np.any(np.iscomplex(L)):
            L = np.real(L)
        rdl = (R, D, L)

    if use_diagonalization:
        for i in range(n_t):
            f[i] = time_correlation_by_diagonalization(P, pi, obs1, obs2, times[i], rdl)
    else:
        start_values = None
        for i in range(n_t):
            f[i], start_values = \
                time_correlation_direct_by_mtx_vec_prod(P, pi, obs1, obs2,
                                                        times[i], start_values, True)
    return f


def time_relaxation_direct_by_mtx_vec_prod(P, p0, obs, time=1, start_values=None, return_pP_k=False):
    r"""Compute time-relaxations of obs with respect of given initial distribution.
    relaxation(k) = p0 P^k obs
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    p0 : ndarray, shape=(n)
        initial distribution
    obs : ndarray, shape=(n)
        Vector representing observable on discrete states.
    time : int or array like
        time point at which the (auto)correlation will be evaluated.
    start_values = (time,
    Returns
    -------
    relaxation : float
    """
    # input checks
    if not type(time) == int:
        if not type(time) == np.int64:
            raise TypeError("given time (%s) is not an integer, but has type: %s"
                            % (str(time), type(time)))
    if obs.shape[0] != P.shape[0]:
        raise ValueError("observable shape not compatible with given matrix")
    if p0.shape[0] != P.shape[0]:
        raise ValueError("shape of init dist p0 (%s) not compatible with given matrix (shape=%s)"
                         % (p0.shape[0], P.shape))
    # propagate in time
    if start_values:  # begin with a previous calculated val
        pk_i = start_values[1]
        time_prev = start_values[0]
        t_diff = time - time_prev
        r = range(t_diff)
    else:
        if time >= 2:
            pk_i = np.dot(np.dot(p0, P), P)  # pk_2
            r = range(time - 2)
        elif time == 1:
            pk_i = np.dot(p0, P)  # propagate once
            r = range(0)
        elif time == 0:  # P^0 = I => p0*I = p0
            pk_i = p0
            r = range(0)

    for k in r:  # perform the rest of the propagations p0 P^t_diff
        pk_i = np.dot(pk_i, P)

    # result
    l = np.dot(pk_i, obs)
    if return_pP_k:
        return l, (time, pk_i)
    else:
        return l


def time_relaxation_direct_by_diagonalization(P, p0, obs, time, rdl=None):
    if rdl is None:
        raise ValueError("no rdl decomposition")
    R, D, L = rdl

    d_times = np.diag(D) ** time
    diag_inds = np.diag_indices_from(D)
    D_time = np.zeros(D.shape, dtype=d_times.dtype)
    D_time[diag_inds] = d_times
    P_time = np.dot(np.dot(R, D_time), L)

    result = np.dot(np.dot(p0, P_time), obs)
    return result


def time_relaxations_direct(P, p0, obs, times=[1]):
    r"""Compute time-relaxations of obs with respect of given initial distribution.
    relaxation(k) = p0 P^k obs
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    p0 : ndarray, shape=(n)
        initial distribution
    obs : ndarray, shape=(n)
        Vector representing observable on discrete states.
    times : array-like, shape(n_t)
        Vector of time points at which the (auto)correlation will be evaluated
    Returns
    -------
    relaxations : ndarray, shape(n_t)
    """
    n_t = len(times)
    times = np.sort(times)

    # maximum time > number of rows?
    if times[-1] > P.shape[0]:
        use_diagonalization = True
        R, D, L = rdl_decomposition(P)
        # discard imaginary part, if all elements i=0
        if not np.any(np.iscomplex(R)):
            R = np.real(R)
        if not np.any(np.iscomplex(D)):
            D = np.real(D)
        if not np.any(np.iscomplex(L)):
            L = np.real(L)
        rdl = (R, D, L)

    f = np.empty(n_t, dtype=D.dtype)

    if use_diagonalization:
        for i in range(n_t):
            f[i] = time_relaxation_direct_by_diagonalization(P, p0, obs, times[i], rdl)
    else:
        start_values = None
        for i in range(n_t):
            f[i], start_values = time_relaxation_direct_by_mtx_vec_prod(P, p0, obs, times[i], start_values, True)
    return f


# p = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]
# walk1 = [2, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 0, 1, 2, 2]
# walk2 = None
# tp = [1, 10]
# mc = MarkovChain(p)
#
# observations1 = np.zeros(mc.size, dtype=float)
#
# for state in walk1:
#     observations1[state] += 1.0
#
# if walk2 is None:
#     observations2 = np.copy(observations1)
# else:
#
#     observations2 = np.zeros(mc.size, dtype=int)
#
#     for state in walk2:
#         observations2[state] += 1.0
#
# aaa = time_correlations_direct(mc.p, mc.pi[0], observations1, observations2, tp)
# zzz = mc.time_correlations(walk1, walk2, tp)
#
# z = 1