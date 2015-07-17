'''Locally Linear Embedding for Regression'''

import numpy as np
from scipy.sparse import eye, csr_matrix
from scipy.sparse.csgraph import laplacian
from sklearn.manifold.locally_linear import (
    barycenter_kneighbors_graph, null_space, LocallyLinearEmbedding)
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors


def ller(X, Y, n_neighbors, n_components, reg=1e-3, eigen_solver='auto',
         tol=1e-6, max_iter=100, random_state=None, mu=0.5, gamma=None):
    """
    Locally Linear Embedding for Regression (LLER)

    Parameters
    ----------
    X : ndarray, 2-dimensional
        The data matrix, shape (num_points, num_dims)

    Y : ndarray, 1 or 2-dimensional
        The response matrix, shape (num_points, num_responses).

    n_neighbors : int
        Number of neighbors for kNN graph construction.

    n_components : int
        Number of dimensions for embedding.

    Returns
    -------
    embedding : ndarray, 2-dimensional
        The embedding of X, shape (num_points, n_components)

    lle_error : ?
        TODO write description and type

    ller_error : ?
        TODO write description and type
    """
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if Y.ndim == 1:
        Y = Y[:, None]

    if gamma is None:
        dists = pairwise_distances(Y)
        gamma = 1.0 / np.median(dists)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nbrs.fit(X)
    X = nbrs._fit_X

    Nx, d_in = X.shape
    Ny = Y.shape[0]

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= Nx:
        raise ValueError("n_neighbors must be less than number of points")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if Nx != Ny:
        raise ValueError("X and Y must have same number of points")

    M_sparse = (eigen_solver != 'dense')

    W = barycenter_kneighbors_graph(
        nbrs, n_neighbors=n_neighbors, reg=reg)

    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1

    P = rbf_kernel(Y, gamma=gamma)
    L = laplacian(P, normed=False)
    M /= np.abs(M).max()  # optional scaling step
    L /= np.abs(L).max()
    omega = M + mu * L
    embedding, lle_error = null_space(omega, n_components, k_skip=1,
                                      eigen_solver=eigen_solver, tol=tol,
                                      max_iter=max_iter,
                                      random_state=random_state)
    ller_error = np.trace(embedding.T.dot(L).dot(embedding))
    return embedding, lle_error, ller_error


class LLER(LocallyLinearEmbedding):
    """Scikit-learn compatible class for LLER."""

    def __init__(self, n_neighbors=5, n_components=2, reg=1E-3,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 neighbors_algorithm='auto', random_state=None,
                 mu=0.5, gamma=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.mu = mu
        self.gamma = gamma

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.embedding_

    def fit(self, X, Y):
        self.nbrs_ = NearestNeighbors(self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)
        self.embedding_, self.lle_error_, self.ller_error_ = ller(
            self.nbrs_, Y, self.n_neighbors, self.n_components,
            eigen_solver=self.eigen_solver, tol=self.tol,
            max_iter=self.max_iter, random_state=self.random_state,
            mu=self.mu, gamma=self.gamma)
        return self
