from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)


class LLMProfiling(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self):
        pass

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        from flask import session
        print(session["diversity_perception"])
        print("Selected", selected_items)
        print("Filter out", filter_out_items)

        self._loader.embeddings_df['movieId'] = self._loader.embeddings_df['movieId'].astype(int)

        movie = self._loader.items_df.iloc[selected_items[0]]
        print(movie)

        input_text = "Genres: "+ ", ".join(movie['genres']) + "\nPlot:" + movie['plot']
        embedding = self.model.encode([input_text])[0]
        embedding = embedding.reshape(1, -1)

        mask = ~self._loader.embeddings_df.index.isin(filter_out_items)
        original_indices = self._loader.embeddings_df[mask].index
        filtered_df = self._loader.embeddings_df[mask]
        filtered_df.reset_index(drop=True, inplace=True)

        emb_matrix = filtered_df.drop(columns=['movieId']).values

        similarities = cosine_similarity(embedding, emb_matrix)[0]

        top_k_sim = np.argsort(similarities)[::-1][:k]

        print(top_k_sim)

        top_k_original_indices = original_indices[top_k_sim].values

        result = [int(i) for i in top_k_original_indices]

        print("Final items (original indices):", result)

        return result

    @classmethod
    def name(cls):
        return "LLMProfiling"

    @classmethod
    def parameters(cls):
        return [
            
        ]