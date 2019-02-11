import functools
import heapq
import logging
import time
import tqdm

import numpy as np
import scipy
import scipy.sparse

import alsi.cuda

from . import _als
from .recommender_base import MatrixFactorizationBase
from .utils import check_blas_config, nonzeros

log = logging.getLogger("alsi")

class AlternatingLeastSquares(MatrixFactorizationBase):

    def __init__(self, factors=100, regularization=0.01, dtype=np.float32,
                 use_native=True, use_cg=True, use_gpu=alsi.cuda.HAS_CUDA,
                 iterations=15, calculate_training_loss=False, num_threads=0):
        super(AlternatingLeastSquares, self).__init__()

        if use_gpu and factors % 32:
            padding = 32 - factors % 32
            log.warning("GPU training requires factor size to be a multiple of 32."
                        " Increasing factors from %i to %i.", factors, factors + padding)
            factors += padding

        self.factors                 = factors
        self.regularization          = regularization
        self.dtype                   = dtype
        self.use_native              = use_native
        self.use_cg                  = use_cg
        self.use_gpu                 = use_gpu
        self.iterations              = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads             = num_threads
        self.fit_callback            = None
        self.cg_steps                = 3
        self._YtY                    = None

        check_blas_config()

    def fit(self, item_users, show_progress=True):

        Ciu = item_users
        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            s = time.time()
            log.debug("Converting input to CSR format")
            Ciu = Ciu.tocsr()
            log.debug("Converted input to CSR in %.3fs", time.time() - s)

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)

        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        if self.user_factors is None:
            self.user_factors = np.random.rand(users, self.factors).astype(self.dtype) * 0.01
        if self.item_factors is None:
            self.item_factors = np.random.rand(items, self.factors).astype(self.dtype) * 0.01

        log.debug("Initialized factors in %s", time.time() - s)

        self._item_norms = None
        self._YtY = None

        if self.use_gpu:
            return self._fit_gpu(Ciu, Cui, show_progress)

        solver = self.solver

        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm.tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                solver(Cui, self.user_factors, self.item_factors, self.regularization,
                       num_threads=self.num_threads)
                progress.update(.5)

                solver(Ciu, self.item_factors, self.user_factors, self.regularization,
                       num_threads=self.num_threads)
                progress.update(.5)

                if self.calculate_training_loss:
                    loss = _als.calculate_loss(Cui, self.user_factors, self.item_factors,
                                               self.regularization, num_threads=self.num_threads)
                    progress.set_postfix({"loss": loss})

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

    def _fit_gpu(self, Ciu_host, Cui_host, show_progress=True):
        if not implicit.cuda.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        if self.dtype == np.float64:
            log.warning("Factors of dtype float64 aren't supported with gpu fitting. "
                        "Converting factors to float32")
            self.item_factors = self.item_factors.astype(np.float32)
            self.user_factors = self.user_factors.astype(np.float32)

        Ciu = implicit.cuda.CuCSRMatrix(Ciu_host)
        Cui = implicit.cuda.CuCSRMatrix(Cui_host)
        X = implicit.cuda.CuDenseMatrix(self.user_factors.astype(np.float32))
        Y = implicit.cuda.CuDenseMatrix(self.item_factors.astype(np.float32))

        solver = implicit.cuda.CuLeastSquaresSolver(self.factors)
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm.tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                solver.least_squares(Cui, X, Y, self.regularization, self.cg_steps)
                progress.update(.5)
                solver.least_squares(Ciu, Y, X, self.regularization, self.cg_steps)
                progress.update(.5)

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.calculate_training_loss:
                    loss = solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        X.to_host(self.user_factors)
        Y.to_host(self.item_factors)

    def recalculate_user(self, userid, user_items):
        return user_factor(self.item_factors, self.YtY,
                           user_items.tocsr(), userid,
                           self.regularization, self.factors)

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):

        user_items = user_items.tocsr()
        if user_weights is None:
            A, _ = user_linear_equation(self.item_factors, self.YtY,
                                        user_items, userid,
                                        self.regularization, self.factors)
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        for i, (itemid, confidence) in enumerate(nonzeros(user_items, userid)):
            if confidence < 0:
                continue

            factor = self.item_factors[itemid]
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, itemid)
            if i < N:
                heapq.heappush(h, contribution)
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

    @property
    def solver(self):
        if self.use_cg:
            solver = _als.least_squares_cg if self.use_native else least_squares_cg
            return functools.partial(solver, cg_steps=self.cg_steps)
        return _als.least_squares if self.use_native else least_squares

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY


def alternating_least_squares(Ciu, factors, **kwargs):
    log.warning("This method is deprecated. Please use the AlternatingLeastSquares"
                " class instead")

    model = AlternatingLeastSquares(factors=factors, **kwargs)
    model.fit(Ciu)
    return model.item_factors, model.user_factors


def least_squares(Cui, X, Y, regularization, num_threads=0):
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        X[u] = user_factor(Y, YtY, Cui, u, regularization, n_factors)


def user_linear_equation(Y, YtY, Cui, u, regularization, n_factors):

    A = YtY + regularization * np.eye(n_factors)
    b = np.zeros(n_factors)

    for i, confidence in nonzeros(Cui, u):
        factor = Y[i]

        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b


def user_factor(Y, YtY, Cui, u, regularization, n_factors):
    A, b = user_linear_equation(Y, YtY, Cui, u, regularization, n_factors)
    return np.linalg.solve(A, b)


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    users, factors = X.shape
    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            if confidence > 0:
                r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]
            else:
                confidence *= -1
                r += - (confidence - 1) * Y[i].dot(x) * Y[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for it in range(cg_steps):

            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x