from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from ollama import chat
import numpy as np
import textwrap
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

        self._hdbscan_clusterer = HDBSCAN(  # TODO: Maybe move this to predict, and set min_cluster_size according to the number of selected items
            min_cluster_size=2,
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

        MAX_CLUSTERS = 3

        ratings = session["diversity_perception"]

        def rating_to_effect(avg_rating):
            return (avg_rating - 3) / 2.0 # rating to [-1, 1] range
        
        def compute_weighted_effect(ratings_data, version_key, sim_key):
            items = ratings_data.get(version_key, [])
            if not items:
                return 0.0
            total = 0.0
            for item in items:
                rating = int(item["rating"])
                sim = float(item[sim_key])
                effect = rating_to_effect(rating)
                total += sim * effect
            return total / len(items)

        plot_effect = -compute_weighted_effect(ratings, "no_div_plot", "plot_sim") \
                      +compute_weighted_effect(ratings, "no_div_genres", "plot_sim")

        genre_effect = +compute_weighted_effect(ratings, "no_div_plot", "genre_sim") \
                       -compute_weighted_effect(ratings, "no_div_genres", "genre_sim")

        # Scale to weights in [0.5, 2.0]
        def scale_to_weight(effect, min_val=0.5, max_val=2.0, neutral_value=0):
            if effect == neutral_value:
                return 1.0
            # Effect to [-1, 1] and normalize to [0.5, 2.0]
            effect = max(-1.0, min(1.0, effect))
            return min_val + (effect + 1) * (max_val - min_val) / 2.0
        # Scale effects to weights in [0.5, 2.0] for embeddings
        plot_weight = scale_to_weight(plot_effect)
        genre_weight = scale_to_weight(genre_effect)

        print(f"Genre weight: {genre_weight:.2f}")
        print(f"Plot weight: {plot_weight:.2f}")

        # Prepare user-preferred movies based on selected items
        user_preferred_movies = []
        for item in selected_items:
            user_preferred_movies.append(self._loader.items_df.iloc[item])

        # print("PREF MOVIES:", user_preferred_movies)

        # Update the final embedding calculation with the genre and plot weights
        final_embedding = genre_weight * self._loader.genres_embeddings + plot_weight * self._loader.plot_embeddings
        print("final embed shape", final_embedding.shape)
        
        mask = np.ones(final_embedding.shape[0], dtype=bool)
        mask[filter_out_items] = False
        original_indices = np.where(mask)[0]
        emb_matrix = final_embedding[mask]
        
        user_genre_embeddings = self._model.encode(["Genres: " + movie['genres'] for movie in user_preferred_movies])
        user_plot_embeddings = self._model.encode(["Plot: " + movie['plot'] for movie in user_preferred_movies])
        user_embeddings = (genre_weight * user_genre_embeddings + plot_weight * user_plot_embeddings) / 2
        print(user_embeddings.shape)

        print("clustering")
        
        cluster_labels = self._hdbscan_clusterer.fit_predict(user_embeddings)
        print("clustering DONE")
        print(cluster_labels)
        clusters = {}

        # Check if clusters have been found
        if len(np.unique(cluster_labels)) == 1:
            random_indices = np.random.choice(len(user_preferred_movies), size=min(len(user_preferred_movies),MAX_CLUSTERS), replace=False)
            for i in random_indices:
                label = "random_" + str(i)
                clusters[label] = user_preferred_movies[i]

        else:
            # Get at most 3 largest clusters, igrnoring noise
            labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
            top_clusters = labels[np.argsort(-counts)[:MAX_CLUSTERS]]
            print("Top clusters:", top_clusters)

            mask = ~np.isin(cluster_labels, top_clusters)
            print(cluster_labels)
            cluster_labels[mask] = -1
            print(cluster_labels)

            for i in range(len(user_preferred_movies)):
                label = cluster_labels[i]
                movie_info = user_preferred_movies[i]
                if label == -1:
                    continue # Skip noise movies
                if label not in clusters:
                    clusters[str(label)] = []
                print(f"Adding movie {movie_info['title']} to cluster {label}")
                clusters[str(label)].append(movie_info)

        representant_embeddings_dict = {}
        representants = []

        print("\n--- Generating Cluster Representants ---")
        for cluster_label, cluster_data in clusters.items():
            print(f"Generating representant for Cluster {cluster_label}...")
            if cluster_label.startswith("random_"):
                representant = Representant(genres=cluster_data["genres"], plot=cluster_data["plot"])
            else:
                representant = self._generate_representant([m for m in cluster_data])
            if representant:
                print(f"Representant {cluster_label}:", representant)

                representants.append(representant)
            
                rep_text = f"Genres: {representant.genres} Plot: {representant.plot}"
                emb = self._model.encode([rep_text])
                representant_embeddings_dict[cluster_label] = emb[0]
            else:
                print(f"Could not generate representant for cluster: {cluster_label}")
        
        print("\n--- Generating Diversity Representant ---")
        div_representant = self._generate_diversity_representant(representants)
        if div_representant:
            print(f"Generated diversity representant:", div_representant)
            rep_genre_embeddings = self._model.encode(["Genres: " + div_representant.genres])
            rep_plot_embeddings = self._model.encode(["Plot: " + div_representant.plot])
            rep_embeddings = (genre_weight * rep_genre_embeddings + plot_weight * rep_plot_embeddings) / 2
            representant_embeddings_dict["diversity"] = rep_embeddings[0]
        else:
            print("Could not generate diversity representant.")

        # Find similar embeddings, movies
        used_items = set()
        cluster_candidates = {}
        for cluster_id, rep_emb in representant_embeddings_dict.items():
            similarities = cosine_similarity(rep_emb.reshape(1, -1), emb_matrix)[0]
            closest_indices = np.argsort(-similarities)[:k]
            top_k_original_indices = original_indices[closest_indices]
            cluster_candidates[cluster_id] = [int(i) for i in top_k_original_indices if int(i) not in used_items]

        # Create result in round-robin way
        # TODO: Use LLM to re-rank the items ? or half of them ?
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

    def _generate_representant(self, movies_cluster):
            movies = "\n".join(f"- {movie['title']} | Genres: " + movie['genres'] + "Plot: " + movie["plot"] for movie in movies_cluster)
            
            llama_prompt = textwrap.dedent(f"""\
                The user enjoys the following movies:
                
                {movies}
                
                Based on these, write a synthetic movie-style entry that represents the user's movie taste.
                Return a short JSON with 'genres' and 'plot'. Genres should be comma-separated keywords. The plot should match the tone and structure of actual movie plots in the dataset.
                """)

            response = chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates user preference profiles for movie recommendation systems. You always return output in a JSON format that matches movie metadata: genres and plot."},
                    {"role": "user", "content": llama_prompt}
                ],
                model="llama3.1",
                format=Representant.model_json_schema(),
            )
            representant = Representant.model_validate_json(response.message.content)

            return representant
    
    def _generate_diversity_representant(self, representants):
        reps = "\n".join(
            f"- Genres: {r.genres} | Plot: {r.plot}"
            for r in representants
        )
        
        llama_prompt = textwrap.dedent(f"""\
            Below are several representants, each summarizing a group of movies the user enjoys:

            {reps}

            Your task:
            1. Choose **genres NOT listed** in any of them.
            2. Based on the selected new genres, create a synthetic 'representant'.
            3. The **plot must match the new genres** and be **different** from the plots above.

            Return a JSON object with:
            - 'genres': a comma-separated string of the new genres
            - 'plot': a short movie-style generated description that fits for new genres and avoids similarities with the input plots
            """)

        response = chat(
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent("""\
                        You are a creative assistant that generates diversity-focused movie profiles.
                        You're given several 'representants' that summarize the user's known movie preferences.
                        Your task is to expand their profile by introducing a completely different angle.
                        Instructions:
                            - Choose only genres that do NOT appear in the input.
                            - Then, write a new movie plot that matches the newly chosen genres.
                            - The plot must NOT share similarities with any of the provided plots in the input.
                                               
                        Your goal is to generate a believable but distinctly different 'representant'.
                        """)
                },
                {"role": "user", "content": llama_prompt}
            ],
            model="llama3.1:8b",
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