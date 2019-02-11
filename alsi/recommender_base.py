
import itertools
from abc import ABCMeta, abstractmethod
import numpy as np

class RecommenderBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, item_users):
        pass

    @abstractmethod
    def recommend(self, userid, user_items,
                  N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False):
        pass

    @abstractmethod
    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        pass

    @abstractmethod
    def similar_users(self, userid, N=10):
        pass

    @abstractmethod
    def similar_items(self, itemid, N=10):
        pass

class MatrixFactorizationBase(RecommenderBase):
    
    def __init__(self):

        self.item_factors = None
        self.user_factors = None
        self._user_norms, self._item_norms = None, None

    def recommend(self, userid, user_items,
                  N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False):
        user = self._user_factor(userid, user_items, recalculate_user)

        if filter_already_liked_items is True:
            liked = set(user_items[userid].indices)
        else:
            liked = set()
        scores = self.item_factors.dot(user)
        if filter_items:
            liked.update(filter_items)

        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        user = self._user_factor(userid, user_items, recalculate_user)

        if max(selected_items) >= user_items.shape[1] or min(selected_items) < 0:
            raise IndexError("Some of selected itemids are not in the model")

        item_factors = self.item_factors[selected_items]
        scores = item_factors.dot(user)

        return sorted(zip(selected_items, scores), key=lambda x: -x[1])

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if recalculate_user:
            return self.recalculate_user(userid, user_items)
        else:
            return self.user_factors[userid]

    def recalculate_user(self, userid, user_items):
        raise NotImplementedError("recalculate_user is not supported with this model")

    def similar_users(self, userid, N=10):
        factor = self.user_factors[userid]
        factors = self.user_factors
        norms = self.user_norms

        return self._get_similarity_score(factor, factors, norms, N)

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(self, itemid, N=10):
        factor = self.item_factors[itemid]
        factors = self.item_factors
        norms = self.item_norms

        return self._get_similarity_score(factor, factors, norms, N)

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

    def _get_similarity_score(self, factor, factors, norms, N):
        scores = factors.dot(factor) / norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = np.linalg.norm(self.user_factors, axis=-1)
            self._user_norms[self._user_norms == 0] = 1e-10
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=-1)
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms
