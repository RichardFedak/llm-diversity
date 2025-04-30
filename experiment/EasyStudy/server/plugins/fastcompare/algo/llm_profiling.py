from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)


class LLMProfiling(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

    def fit(self):
        pass

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        from flask import session
        print(session["diversity_perception"])
        print(self._loader.embeddings_df.iloc[0])
        print(self._loader.ratings_df.iloc[0])
        print(self._loader.embeddings_df.movieId.dtype)
        print(self._loader.embeddings_df.shape)

        self._loader.embeddings_df['movieId'] = self._loader.embeddings_df['movieId'].astype(int)

        ref_row = self._loader.embeddings_df.iloc[0]
        ref_id = ref_row['movieId']
        ref_embedding = ref_row.drop('movieId').values.reshape(1, -1)

        ref_row = self._loader.embeddings_df.iloc[0]
        ref_id = ref_row['movieId']
        ref_embedding = ref_row.drop('movieId').values.reshape(1, -1)

        mask = ~self._loader.embeddings_df['movieId'].isin(filter_out_items + [ref_id])
        filtered_df = self._loader.embeddings_df[mask]
        filtered_indices = filtered_df.index  # Original row indices

        emb_matrix = filtered_df.drop(columns=['movieId']).values

        similarities = cosine_similarity(ref_embedding, emb_matrix)[0]

        top_k_indices = [filtered_indices[i] for i in np.argsort(similarities)[::-1][:k]]

        result = [int(i) for i in top_k_indices]

        return result

    @classmethod
    def name(cls):
        return "LLMProfiling"

    @classmethod
    def parameters(cls):
        return [
            
        ]