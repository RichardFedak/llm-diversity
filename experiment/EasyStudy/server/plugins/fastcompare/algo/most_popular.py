from abc import ABC

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)


class MostPopular(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

    def fit(self, _):
        pass

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k, weights, items_count, div_perception):
        ratings = self._ratings_df.groupby("item").size().sort_values(ascending=False)

        result = ratings.index.values.tolist()

        result = [r for r in result if r not in filter_out_items][:k]

        return result

    @classmethod
    def name(cls):
        return "MostPopular"

    @classmethod
    def parameters(cls):
        return [
            
        ]