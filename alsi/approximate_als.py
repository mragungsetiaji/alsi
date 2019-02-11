
import itertools
import logging
import numpy

from alsi.als import AlternatingLeastSquares
import alsi.cuda

log = logging.getLogger("alsi")

def augment_inner_product_matrix(factors):

    norms = numpy.linalg.norm(factors, axis=1)
    max_norm = norms.max()
    extra_dimension = numpy.sqrt(max_norm ** 2 - norms ** 2)
    return max_norm, numpy.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)


class NMSLibAlternatingLeastSquares(AlternatingLeastSquares):

    def __init__(self,
                 approximate_similar_items=True, approximate_recommend=True,
                 method='hnsw', index_params=None, query_params=None, *args, **kwargs):
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}
        if query_params is None:
            query_params = {'ef': 90}

        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend
        self.method = method

        self.index_params = index_params
        self.query_params = query_params

        super(NMSLibAlternatingLeastSquares, self).__init__(*args, **kwargs)

    def fit(self, Ciu, show_progress=True):

        logging.getLogger('nmslib').setLevel(logging.WARNING)
        import nmslib

        super(NMSLibAlternatingLeastSquares, self).fit(Ciu, show_progress)

        if self.approximate_similar_items:
            log.debug("Building nmslib similar items index")
            self.similar_items_index = nmslib.init(
                method=self.method, space='cosinesimil')

            norms = numpy.linalg.norm(self.item_factors, axis=1)
            ids = numpy.arange(self.item_factors.shape[0])

            item_factors = numpy.delete(self.item_factors, ids[norms == 0], axis=0)
            ids = ids[norms != 0]

            self.similar_items_index.addDataPointBatch(item_factors, ids=ids)
            self.similar_items_index.createIndex(self.index_params,
                                                 print_progress=show_progress)
            self.similar_items_index.setQueryTimeParams(self.query_params)

        if self.approximate_recommend:
            log.debug("Building nmslib recommendation index")
            self.max_norm, extra = augment_inner_product_matrix(
                self.item_factors)
            self.recommend_index = nmslib.init(
                method='hnsw', space='cosinesimil')
            self.recommend_index.addDataPointBatch(extra)
            self.recommend_index.createIndex(self.index_params, print_progress=show_progress)
            self.recommend_index.setQueryTimeParams(self.query_params)

    def similar_items(self, itemid, N=10):
        if not self.approximate_similar_items:
            return super(NMSLibAlternatingLeastSquares, self).similar_items(itemid, N)

        neighbours, distances = self.similar_items_index.knnQuery(
            self.item_factors[itemid], N)
        return zip(neighbours, 1.0 - distances)

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        if not self.approximate_recommend:
            return super(NMSLibAlternatingLeastSquares,
                         self).recommend(userid, user_items, N=N,
                                         filter_items=filter_items,
                                         recalculate_user=recalculate_user)

        user = self._user_factor(userid, user_items, recalculate_user)

        liked = set(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)
        count = N + len(liked)

        query = numpy.append(user, 0)
        ids, dist = self.recommend_index.knnQuery(query, count)

        scaling = self.max_norm * numpy.linalg.norm(query)
        dist = scaling * (1.0 - dist)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in liked), N))


class AnnoyAlternatingLeastSquares(AlternatingLeastSquares):

    def __init__(self, approximate_similar_items=True, approximate_recommend=True,
                 n_trees=50, search_k=-1, *args, **kwargs):
        super(AnnoyAlternatingLeastSquares, self).__init__(*args, **kwargs)
        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        self.n_trees = n_trees
        self.search_k = search_k

    def fit(self, Ciu, show_progress=True):
    
        import annoy

        super(AnnoyAlternatingLeastSquares, self).fit(Ciu, show_progress)

        if self.approximate_similar_items:
            log.debug("Building annoy similar items index")

            self.similar_items_index = annoy.AnnoyIndex(
                self.item_factors.shape[1], 'angular')
            for i, row in enumerate(self.item_factors):
                self.similar_items_index.add_item(i, row)
            self.similar_items_index.build(self.n_trees)

        if self.approximate_recommend:
            log.debug("Building annoy recommendation index")
            self.max_norm, extra = augment_inner_product_matrix(self.item_factors)
            self.recommend_index = annoy.AnnoyIndex(extra.shape[1], 'angular')
            for i, row in enumerate(extra):
                self.recommend_index.add_item(i, row)
            self.recommend_index.build(self.n_trees)

    def similar_items(self, itemid, N=10):
        if not self.approximate_similar_items:
            return super(AnnoyAlternatingLeastSquares, self).similar_items(itemid, N)

        neighbours, dist = self.similar_items_index.get_nns_by_item(itemid, N,
                                                                    search_k=self.search_k,
                                                                    include_distances=True)
        return zip(neighbours, 1 - (numpy.array(dist) ** 2) / 2)

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        if not self.approximate_recommend:
            return super(AnnoyAlternatingLeastSquares,
                         self).recommend(userid, user_items, N=N,
                                         filter_items=filter_items,
                                         recalculate_user=recalculate_user)

        user = self._user_factor(userid, user_items, recalculate_user)

        liked = set(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)
        count = N + len(liked)

        query = numpy.append(user, 0)
        ids, dist = self.recommend_index.get_nns_by_vector(query, count, include_distances=True,
                                                           search_k=self.search_k)

        scaling = self.max_norm * numpy.linalg.norm(query)
        dist = scaling * (1 - (numpy.array(dist) ** 2) / 2)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in liked), N))


class FaissAlternatingLeastSquares(AlternatingLeastSquares):

    def __init__(self, approximate_similar_items=True, approximate_recommend=True,
                 nlist=400, nprobe=20, use_gpu=implicit.cuda.HAS_CUDA, *args, **kwargs):
        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        # hyper-parameters for FAISS
        self.nlist = nlist
        self.nprobe = nprobe
        super(FaissAlternatingLeastSquares, self).__init__(*args, use_gpu=use_gpu, **kwargs)

    def fit(self, Ciu, show_progress=True):
        import faiss

        # train the model
        super(FaissAlternatingLeastSquares, self).fit(Ciu, show_progress)

        self.quantizer = faiss.IndexFlat(self.factors)

        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        item_factors = self.item_factors.astype('float32')

        if self.approximate_recommend:
            log.debug("Building faiss recommendation index")

            # build up a inner product index here
            if self.use_gpu:
                index = faiss.GpuIndexIVFFlat(self.gpu_resources, self.factors, self.nlist,
                                              faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFFlat(self.quantizer, self.factors, self.nlist,
                                           faiss.METRIC_INNER_PRODUCT)

            index.train(item_factors)
            index.add(item_factors)
            index.nprobe = self.nprobe
            self.recommend_index = index

        if self.approximate_similar_items:
            log.debug("Building faiss similar items index")

            norms = numpy.linalg.norm(item_factors, axis=1)
            norms[norms == 0] = 1e-10

            normalized = (item_factors.T / norms).T.astype('float32')
            if self.use_gpu:
                index = faiss.GpuIndexIVFFlat(self.gpu_resources, self.factors, self.nlist,
                                              faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFFlat(self.quantizer, self.factors, self.nlist,
                                           faiss.METRIC_INNER_PRODUCT)

            index.train(normalized)
            index.add(normalized)
            index.nprobe = self.nprobe
            self.similar_items_index = index

    def similar_items(self, itemid, N=10):
        if not self.approximate_similar_items or (self.use_gpu and N >= 1024):
            return super(FaissAlternatingLeastSquares, self).similar_items(itemid, N)

        factors = self.item_factors[itemid]
        factors /= numpy.linalg.norm(factors)
        (dist,), (ids,) = self.similar_items_index.search(factors.reshape(1, -1).astype('float32'),
                                                          N)
        return zip(ids, dist)

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        if not self.approximate_recommend:
            return super(FaissAlternatingLeastSquares,
                         self).recommend(userid, user_items, N=N,
                                         filter_items=filter_items,
                                         recalculate_user=recalculate_user)

        user = self._user_factor(userid, user_items, recalculate_user)

        liked = set(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)
        count = N + len(liked)

        if self.use_gpu and count >= 1024:
            return super(FaissAlternatingLeastSquares,
                         self).recommend(userid, user_items, N=N,
                                         filter_items=filter_items,
                                         recalculate_user=recalculate_user)

        query = user.reshape(1, -1).astype('float32')
        (dist,), (ids,) = self.recommend_index.search(query, count)
        return list(itertools.islice((rec for rec in zip(ids, dist) if rec[0] not in liked), N))
