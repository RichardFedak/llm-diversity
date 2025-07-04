from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from ollama import chat
import numpy as np
import textwrap
from typing import List
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)

class SelectedMovie(BaseModel):
    item_id: int
    reason: str

class SelectionResult(BaseModel):
    selected: List[SelectedMovie]

from plugins.fastcompare.algo.representant_base import (
    Representant,
    DiversityStimulus,
    RepresentantGenerator,
    GenresDiversityHandler,
    PlotDiversityHandler,
)

class LLMProfilingDiv(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = None
        self._loader = None

        self._model = None

        self._hdbscan_clusterer = None

        self.stimulus_handlers: dict[DiversityStimulus, type[RepresentantGenerator]] = {
            DiversityStimulus.GENRES: GenresDiversityHandler,
            DiversityStimulus.PLOT: PlotDiversityHandler,
        }

        self.diversity_stimulus = None

    def fit(self, loader):
        self._ratings_df = loader.ratings_df
        self._loader = loader

        self._model = SentenceTransformer('all-MiniLM-L6-v2')

        self._hdbscan_clusterer = HDBSCAN(
            min_cluster_size=2,
            min_samples=None,
            metric='cosine',
        )

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k, weights, items_count, div_perception):
        #print("Selected", selected_items)
        #print("Filter out", filter_out_items)

        MAX_CLUSTERS = 3

        ratings = div_perception

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

        self.diversity_stimulus = DiversityStimulus.GENRES if genre_weight > plot_weight else DiversityStimulus.PLOT

        #print(f"Genre weight: {genre_weight:.2f}")
        #print(f"Plot weight: {plot_weight:.2f}")

        # Prepare user-preferred movies based on selected items
        user_preferred_movies = []
        for item in selected_items:
            user_preferred_movies.append(self._loader.items_df.iloc[item])

        # #print("PREF MOVIES:", user_preferred_movies)

        # Update the final embedding calculation with the genre and plot weights
        final_embedding = genre_weight * self._loader.genres_embeddings + plot_weight * self._loader.plot_embeddings
        #print("final embed shape", final_embedding.shape)
        
        mask = np.ones(final_embedding.shape[0], dtype=bool)
        mask[filter_out_items] = False
        original_indices = np.where(mask)[0]
        emb_matrix = final_embedding[mask]
        
        user_genre_embeddings = self._model.encode(["Genres: " + movie['genres'] for movie in user_preferred_movies])
        user_plot_embeddings = self._model.encode(["Plot: " + movie['plot'] for movie in user_preferred_movies])
        user_embeddings = (genre_weight * user_genre_embeddings + plot_weight * user_plot_embeddings) / 2
        #print(user_embeddings.shape)

        #print("clustering")
        
        cluster_labels = self._hdbscan_clusterer.fit_predict(user_embeddings)
        #print("clustering DONE")
        #print(cluster_labels)
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
            #print("Top clusters:", top_clusters)

            mask = ~np.isin(cluster_labels, top_clusters)
            #print(cluster_labels)
            cluster_labels[mask] = -1
            #print(cluster_labels)

            for i in range(len(user_preferred_movies)):
                label = cluster_labels[i]
                movie_info = user_preferred_movies[i]
                if label == -1:
                    continue # Skip noise movies
                if label not in clusters:
                    clusters[str(label)] = []
                #print(f"Adding movie {movie_info['title']} to cluster {label}")
                clusters[str(label)].append(movie_info)

        tasks = list(clusters.items())

        def _produce(label, data):
            """Runs *inside* a worker thread — keep heavy code here."""
            if label.startswith("random_"):
                rep = Representant(genres=data["genres"], plot=data["plot"])
            else:
                # potentially slow call
                rep = self._generate_representant(list(data), self.diversity_stimulus)

            if not rep:                         # return sentinel on failure
                return label, None, None

            rep_text = f"Genres: {rep.genres} Plot: {rep.plot}"
            emb_vec = self._model.encode([rep_text])[0]

            return label, rep, emb_vec

        representants                 = []
        representant_embeddings_dict  = {}

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(_produce, lbl, data) for lbl, data in tasks]

            for fut in as_completed(futures):
                label, rep, emb = fut.result()
                if rep is not None:
                    representants.append(rep)
                    representant_embeddings_dict[label] = emb
        
        #print("\n--- Generating Diversity Representant ---")
        div_representant = self._generate_diversity_representant(representants, self.diversity_stimulus)
        if div_representant:
            #print(f"Generated diversity representant:", div_representant)
            rep_genre_embeddings = self._model.encode(["Genres: " + div_representant.genres])
            rep_plot_embeddings = self._model.encode(["Plot: " + div_representant.plot])
            rep_embeddings = (genre_weight * rep_genre_embeddings + plot_weight * rep_plot_embeddings) / 2
            representant_embeddings_dict["diversity"] = rep_embeddings[0]
        else:
            #print("Could not generate diversity representant.")
            pass

        # Find similar embeddings, movies
        used_items = set()
        cluster_candidates = {}
        for cluster_id, rep_emb in representant_embeddings_dict.items():
            similarities = cosine_similarity(rep_emb.reshape(1, -1), emb_matrix)[0]
            closest_indices = np.argsort(-similarities)[:k]
            top_k_original_indices = original_indices[closest_indices]
            cluster_candidates[cluster_id] = [int(i) for i in top_k_original_indices if int(i) not in used_items]
        
        all_candidates = [item for candidates in cluster_candidates.values() for item in candidates]
        all_candidates_data = []
        for item_id in all_candidates:
            row = self._loader.items_df.iloc[item_id]
            all_candidates_data.append((item_id, row))

        res = self.select_diverse_movies(self.diversity_stimulus, all_candidates_data, k)

        cluster_map = {}
        for cluster_id, item_ids in cluster_candidates.items():
            for item_id in item_ids:
                cluster_map[item_id] = cluster_id

        final_ids = [entry.item_id for entry in res.selected][:k]

        cluster_counts = {}
        for item_id in final_ids:
            cluster_id = cluster_map.get(item_id)
            if cluster_id is not None:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        #print("Cluster counts:", cluster_counts)

        return final_ids
    
    def select_diverse_movies(self, stimulus , movie_data: list[dict], k: int):
        if stimulus == DiversityStimulus.GENRES:
            return self.select_genre_diverse_movies(movie_data, k)
        elif stimulus == DiversityStimulus.PLOT:
            return self.select_plot_diverse_movies(movie_data, k)
        else: # should never happen...
            raise ValueError(f"Unknown diversity stimulus: {stimulus}")

    def select_genre_diverse_movies(self, movie_data: list[dict], k: int):
        movie_lines = "\n".join(
            f"- ID: {item[0]} | Title: {item[1]['title']} | Genres: {item[1]['genres']}"
            for item in movie_data
        )

        user_prompt = textwrap.dedent(f"""\
            Here is a list of movies, each with an ID, title, and genres:

            {movie_lines}

            Task:
            - Select {k} movies that are the most diverse from each other based on **genres**.
            - You should aim for the widest genre coverage and minimal overlap between selected movies.
            - Avoid picking movies that are genre-similar to each other.

            Return your answer as a JSON object with a key:
            - 'selected': a list of objects, each containing:
                - 'item_id': the ID of the selected movie
                - 'reason': a short explanation for why it was chosen (e.g., unique genres, expands diversity, etc.)

            Eaxmple structure:
            {{
                "selected": [
                    {{"item_id": 123, "reason": "Your reasoning..."}}
                    ...
                ]
            }}
        """)

        system_prompt = textwrap.dedent("""\
            You are a movie selection assistant helping to maximize diversity in genre across a small list of chosen movies.
            Your job is to pick the most genre-different items from a given list and explain your selection.
            Output must always be in valid JSON format.
        """)

        response = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3.1:8b",
            format=SelectionResult.model_json_schema(),
        )

        parsed = SelectionResult.model_validate_json(response.message.content)
        return parsed

    def select_plot_diverse_movies(self, movie_data: list[dict], k: int):
        movie_lines = "\n".join(
            f"- ID: {item[0]} | Title: {item[1]['title']} | Plot: {item[1]['plot']}"
            for item in movie_data
        )

        user_prompt = textwrap.dedent(f"""\  
            Here is a list of movies, each with an ID, title, and plot description:

            {movie_lines}

            Task:
            - Select {k} movies that are the most diverse from each other based on **plot**.
            - You should aim for the widest thematic and narrative variety, avoiding movies with similar storylines, characters, or settings.

            Return your answer as a JSON object with a key:
            - 'selected': a list of objects, each containing:
                - 'item_id': the ID of the selected movie
                - 'reason': a short explanation for why it was chosen (e.g., unique story, different narrative style, contrasting themes, etc.)
            
            Example structure:
            {{
                "selected": [
                    {{"item_id": 123, "reason": "Your reasoning..."}}
                    ...
                ]
            }}
        """)

        system_prompt = textwrap.dedent("""\
            You are a movie selection assistant helping to maximize diversity in movie **plots**.
            Your job is to pick the most plot-different items from a given list and explain your selection.
            Focus on selecting movies with distinct stories, themes, tones, or narrative styles.
            Output must always be in valid JSON format.
        """)

        response = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3.1:8b",
            format=SelectionResult.model_json_schema(),
        )

        parsed = SelectionResult.model_validate_json(response.message.content)
        return parsed

    def _generate_representant(self, movies_cluster, stimulus: DiversityStimulus):
            
        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_cluster_representant(movies_cluster)
    
    def _generate_diversity_representant(self, representants, stimulus: DiversityStimulus):

        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_diversity_representant(representants)

    @classmethod
    def name(cls):
        return "LLMProfilingDiv"

    @classmethod
    def parameters(cls):
        return [
            
        ]