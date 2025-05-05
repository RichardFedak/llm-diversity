from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from ollama import chat
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)

class Representant(BaseModel):
        genres: str
        plot: str

class LLMProfiling(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._model = SentenceTransformer('all-MiniLM-L6-v2')

        self._hdbscan_clusterer = HDBSCAN(
            min_cluster_size=3,
            min_samples=None,
            metric='cosine',
        )

    def fit(self):
        pass

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        from flask import session
        print(session["diversity_perception"])
        print("Selected", selected_items)
        print("Filter out", filter_out_items)

        ratings = session["diversity_perception"]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def rating_to_effect(avg_rating):
            return (avg_rating - 3) / 2.0

        plot_effect = -rating_to_effect(avg(ratings.get('no_div_plot', []))) + rating_to_effect(avg(ratings.get('no_div_genres', [])))
        genre_effect = +rating_to_effect(avg(ratings.get('no_div_plot', []))) - rating_to_effect(avg(ratings.get('no_div_genres', [])))

        # Scale to weights in [0.5, 2.0]
        def scale_to_weight(effect, min_val=0.5, max_val=2.0, neutral_value=0):
            if effect == neutral_value:
                return 1.0
            # Otherwise, clamp the effect to [-1, 1] and normalize to [0.5, 2.0]
            effect = max(-1.0, min(1.0, effect))
            return min_val + (effect + 1) * (max_val - min_val) / 2.0

        plot_weight = scale_to_weight(plot_effect)
        genre_weight = scale_to_weight(genre_effect)

        print(f"Genre weight: {genre_weight:.2f}")
        print(f"Plot weight: {plot_weight:.2f}")

        # Prepare user-preferred movies based on selected items
        user_preferred_movies = []
        for item in selected_items:
            user_preferred_movies.append(self._loader.items_df.iloc[item])

        # Update the final embedding calculation with the genre and plot weights
        final_embedding = genre_weight * self._loader.genres_embeddings + plot_weight * self._loader.plot_embeddings
        print("final embed shape", final_embedding.shape)
        
        # TODO: CREATE EMBEDDINGS SEPARATELY ? GENRES / PLOT
        mask = np.ones(final_embedding.shape[0], dtype=bool)
        mask[filter_out_items] = False
        original_indices = np.where(mask)[0]
        emb_matrix = final_embedding[mask]

        # mask = ~self._loader.embeddings_df.index.isin(filter_out_items)
        # original_indices = self._loader.embeddings_df[mask].index
        # filtered_df = self._loader.embeddings_df[mask]
        # filtered_df.reset_index(drop=True, inplace=True)

        
        # TODO: USER **MUST** CHOOSE AT LEAST *N* MOVIES
        # TODO: OR .. HANDLE WHEN SELECTED ITEMS < *N*
        print("embeding user")
        user_genre_embeddings = self._model.encode(["Genres: " + movie['genres'] for movie in user_preferred_movies])
        user_plot_embeddings = self._model.encode(["Plot: " + movie['plot'] for movie in user_preferred_movies])
        user_embeddings = (genre_weight * user_genre_embeddings + plot_weight * user_plot_embeddings) / 2
        print("embeding user DONE")
        print(user_embeddings.shape)

        print("clustering")
        
        cluster_labels = self._hdbscan_clusterer.fit_predict(user_embeddings)
        print("clustering DONE")
        print(cluster_labels)

        clusters = {}
        for i in range(len(user_preferred_movies)):
            label = cluster_labels[i]
            movie_info = user_preferred_movies[i]
            if label == -1:
                continue # Skip noise movies
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({**movie_info, "original_index": i})

        representant_embeddings_dict = {}

        print("\n--- Generating Cluster Representants ---")
        for cluster_id, cluster_movies_data in clusters.items():
            print(f"Generating representant for Cluster {cluster_id}...")

            representant = self._generate_representant([m for m in cluster_movies_data])
            if representant:
                print(f"Representant {cluster_id}:", representant)
            
                rep_text = f"Genres: {representant.genres} Plot: {representant.plot}"
                emb = self._model.encode([rep_text])
                representant_embeddings_dict[cluster_id] = emb[0]
            else:
                print(f"Skipping representant for Cluster {cluster_id} due to generation error.")

        # Find similar embeddings, movies
        used_items = set()
        cluster_candidates = {}
        for cluster_id, rep_emb in representant_embeddings_dict.items():
            similarities = cosine_similarity(rep_emb.reshape(1, -1), emb_matrix)[0]
            closest_indices = np.argsort(similarities)[::-1][:k]
            top_k_original_indices = original_indices[closest_indices]
            cluster_candidates[cluster_id] = [int(i) for i in top_k_original_indices if int(i) not in used_items]

        # Create result in round-robin way
        cluster_ids = list(cluster_candidates.keys())
        result = []
        i = 0
        while len(result) < k and any(cluster_candidates.values()):
            cluster_id = cluster_ids[i % len(cluster_ids)]
            candidates = cluster_candidates[cluster_id]

            if candidates:
                candidate = candidates.pop(0)
                if candidate not in used_items:
                    result.append(candidate)
                    used_items.add(candidate)
            i += 1

        return result[:k]

    # TODO: ADD TO PROMPT TO REMOVE POTENTIAL OUTLIERS THAT MAY NOT BE CAUGHT BY HDBSCAN...
    def _generate_representant(self, movies_cluster):
            movies = "\n".join(f"- {movie['title']} | Genres: " + movie['genres'] + "Plot: " + movie["plot"] for movie in movies_cluster)
            
            llama_prompt = f"""
            The user enjoys the following movies:
            
            {movies}
            
            Based on these, write a synthetic movie-style entry that represents the user's movie taste.
            Return a short JSON with 'genres' and 'plot'. Genres should be comma-separated keywords. The plot should match the tone and structure of actual movie plots in the dataset.
            """

            response = chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates user preference profiles for movie recommendation systems. You always return output in a JSON format that matches movie metadata: genres and plot."},
                    {"role": "user", "content": llama_prompt}
                ],
                model="mistral",
                format=Representant.model_json_schema(),
            )
            representant = Representant.model_validate_json(response.message.content)

            return representant

    @classmethod
    def name(cls):
        return "LLMProfiling"

    @classmethod
    def parameters(cls):
        return [
            
        ]