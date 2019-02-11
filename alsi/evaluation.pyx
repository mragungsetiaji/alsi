import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import cython
from cython.operator import dereference
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fmin

from libcpp.unordered_set cimport unordered_set

def train_test_split(ratings, train_percentage=0.8):

    ratings = ratings.tocoo()

    random_index = np.random.random(len(ratings.data))
    train_index = random_index < train_percentage
    test_index = random_index >= train_percentage

    train = csr_matrix((ratings.data[train_index],
                       (ratings.row[train_index], ratings.col[train_index])),
                       shape=ratings.shape, dtype=ratings.dtype)

    test = csr_matrix((ratings.data[test_index],
                      (ratings.row[test_index], ratings.col[test_index])),
                      shape=ratings.shape, dtype=ratings.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test


@cython.boundscheck(False)
def precision_at_k(model, train_user_items, test_user_items, int K=10,
                   show_progress=True, int num_threads=1):

    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], u, i
    cdef double relevant = 0, total = 0
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    progress = tqdm.tqdm(total=users, disable=not show_progress)

    with nogil, parallel(num_threads=num_threads):
        ids = <int *> malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in prange(users, schedule='guided'):

                if test_indptr[u] == test_indptr[u+1]:
                    with gil:
                        progress.update(1)
                    continue
                memset(ids, 0, sizeof(int) * K)

                with gil:
                    recs = model.recommend(u, train_user_items, N=K)
                    for i in range(len(recs)):
                        ids[i] = recs[i][0]
                    progress.update(1)

                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])

                total += fmin(K, likes.size())

                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
        finally:
            free(ids)
            del likes

    progress.close()
    return relevant / total


@cython.boundscheck(False)
def mean_average_precision_at_k(model, train_user_items, test_user_items, int K=10,
                                show_progress=True, int num_threads=1):

    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], u, i, total = 0
    cdef double mean_ap = 0, ap = 0, relevant = 0
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    progress = tqdm.tqdm(total=users, disable=not show_progress)

    with nogil, parallel(num_threads=num_threads):
        ids = <int *> malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in prange(users, schedule='guided'):
                # if we don't have any test items, skip this user
                if test_indptr[u] == test_indptr[u+1]:
                    with gil:
                        progress.update(1)
                    continue
                memset(ids, 0, sizeof(int) * K)

                with gil:
                    recs = model.recommend(u, train_user_items, N=K)
                    for i in range(len(recs)):
                        ids[i] = recs[i][0]
                    progress.update(1)

                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])

                ap = 0
                relevant = 0
                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant = relevant + 1
                        ap = ap + relevant / (i + 1)
                mean_ap += ap / fmin(K, likes.size())
                total += 1
        finally:
            free(ids)
            del likes

    progress.close()
    return mean_ap / total