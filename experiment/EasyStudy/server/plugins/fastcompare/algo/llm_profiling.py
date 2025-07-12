from abc import ABC
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import QuantileTransformer
from ollama import chat
import numpy as np
from typing import List
import time
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

from plugins.fastcompare.algo.representant_base import (
    Representant,
    DiversityStimulus,
    RepresentantGenerator,
    GenresDiversityHandler,
    PlotDiversityHandler,
    get_stimulus_weight
)

class SelectedMovie(BaseModel):
    movie_id: int
    reason: str

class SelectionResult(BaseModel):
    selected: List[SelectedMovie]

class LLMProfiling(AlgorithmBase, ABC):

    def __init__(self, loader, post_processing_method, **kwargs):
        self._ratings_df = None
        self._loader = None
        self._all_items = None

        self._model = None

        self._hdbscan_clusterer = None
        self._max_clusters = None

        self.stimulus_handlers: dict[DiversityStimulus, type[RepresentantGenerator]] = {
            DiversityStimulus.GENRES: GenresDiversityHandler,
            DiversityStimulus.PLOT: PlotDiversityHandler,
        }

        self.diversity_stimulus = None

        self.post_processing_method = post_processing_method

    def fit(self, loader):

        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._model = SentenceTransformer('all-MiniLM-L6-v2')

        self._hdbscan_clusterer = HDBSCAN(
            min_cluster_size=2,
            min_samples=None,
            metric='cosine',
        )

        self._max_clusters = 2

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k, weights, items_count, div_perception):
        indices = list(selected_items)
        user_vector = np.zeros((items_count,), dtype=np.float32)
        for i in indices:
            user_vector[i] = 1.0

        # Calculate EASE relevance scores
        relevance_scores = np.dot(user_vector, weights)

        # Determine diversity stimulus based on user responses from the diversity phase
        ratings = div_perception
        all_pairs = div_perception.get('sim_genres', []) + div_perception.get('sim_plot', [])
        ratings = [int(item['rating']) for item in all_pairs]
        self.diversity_stimulus = self._compute_stimulus(all_pairs, ratings)

        # Prepare user-preferred movies and their metadata, based on IDs
        user_preferred_movies = []
        for item in selected_items:
            user_preferred_movies.append(self._loader.items_df.iloc[item])

        # Prepare weights and embeddings based on the diversity stimulus
        stimulus_weight = get_stimulus_weight(self.diversity_stimulus)
        genre_weight = stimulus_weight if self.diversity_stimulus == DiversityStimulus.GENRES else 1 - stimulus_weight
        plot_weight = 1 - genre_weight
        final_embedding = genre_weight * self._loader.genres_embeddings + plot_weight * self._loader.plot_embeddings

        # Mask already seen items - filter_out_items
        mask = np.ones(final_embedding.shape[0], dtype=bool)
        mask[filter_out_items] = 0
        relevance_scores[filter_out_items] = 0
        original_indices = np.where(mask)[0]
        emb_matrix = final_embedding[mask]
        
        # Generate embdeddings for preferred movies
        user_genre_embeddings = self._model.encode([movie['genres'] for movie in user_preferred_movies])
        user_plot_embeddings = self._model.encode([movie['plot'] for movie in user_preferred_movies])
        user_embeddings = genre_weight * user_genre_embeddings + plot_weight * user_plot_embeddings

        # Cluster user embeddings
        cluster_labels = self._hdbscan_clusterer.fit_predict(user_embeddings)
        clusters = {}

        # Check if clusters have been found
        if len(np.unique(cluster_labels)) == 1: # No clusters found sample randomly
            random_indices = np.random.choice(len(user_preferred_movies), size=min(len(user_preferred_movies),self._max_clusters), replace=False)
            for i in random_indices:
                label = "random_" + str(i)
                clusters[label] = user_preferred_movies[i]
        else:
            labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)

            # Get index of most and least common clusters
            most_common_cluster = labels[np.argmax(counts)]
            least_common_cluster = labels[np.argmin(counts)]

            # If they are the same (clusters have same size), pick a different one
            if most_common_cluster == least_common_cluster and len(labels) > 1:
                # Exclude the most common cluster and pick randomly from the rest
                alternative_clusters = [label for label in labels if label != most_common_cluster]
                least_common_cluster = np.random.choice(alternative_clusters)

            selected_clusters = [most_common_cluster, least_common_cluster]

            cluster_mask = ~np.isin(cluster_labels, selected_clusters)
            cluster_labels[cluster_mask] = -1 # Mask remaining clusters as noise

            for i in range(len(user_preferred_movies)):
                label = cluster_labels[i]
                movie_info = user_preferred_movies[i]
                if label == -1:
                    continue # Skip noise movies
                if str(label) not in clusters:
                    clusters[str(label)] = []
                clusters[str(label)].append(movie_info)

        tasks = list(clusters.items())

        def _produce(label, data):
            """Runs *inside* a worker thread"""
            if label.startswith("random_"): # When we have no clusters, representant is random movie
                rep = Representant(genres=data["genres"], plot=data["plot"])
            else:
                limit = min(len(data), 10)  # Limit to 10 movies per cluster since we are using Llama3.1:8b
                indices = np.random.choice(len(data), size=limit, replace=False)
                cluster_sample = [data[i] for i in indices]
                rep = self._generate_representant(cluster_sample, self.diversity_stimulus)

            if not rep:
                return label, None, None

            rep_genre_embeddings = self._model.encode([rep.genres])
            rep_plot_embeddings = self._model.encode([rep.plot])
            rep_embeddings = genre_weight * rep_genre_embeddings + plot_weight * rep_plot_embeddings

            return label, rep, rep_embeddings

        representants = []
        representant_embeddings_dict = {}

        # Parallelize representant generation
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(_produce, lbl, data) for lbl, data in tasks]

            for fut in as_completed(futures):
                label, rep, emb = fut.result()
                if rep is not None:
                    representants.append(rep)
                    representant_embeddings_dict[label] = emb
        
        div_representant = self._generate_diversity_representant(representants, self.diversity_stimulus)
        if div_representant:
            rep_genre_embeddings = self._model.encode([div_representant.genres])
            rep_plot_embeddings = self._model.encode([div_representant.plot])
            rep_embeddings = genre_weight * rep_genre_embeddings + plot_weight * rep_plot_embeddings
            representant_embeddings_dict["diversity"] = rep_embeddings
        
        # Find similar movies to representants
        cluster_candidates = {}
        for cluster_id, rep_emb in representant_embeddings_dict.items():
            similarities = cosine_similarity(rep_emb.reshape(1, -1), emb_matrix)[0]
            # Normalize similarities and relevance scores
            similarities = self._apply_quantile_transform(similarities.reshape(-1, 1), output_distribution='normal').flatten()
            relevance_scores = self._apply_quantile_transform(relevance_scores.reshape(-1, 1), output_distribution='normal').flatten()

            similarities = similarities * relevance_scores[mask]
            closest_indices = np.argsort(-similarities)[:k]
            top_k_original_indices = original_indices[closest_indices]
            cluster_candidates[cluster_id] = [int(i) for i in top_k_original_indices]

        # Create final list of items based on cluster candidates and selected post-processing method
        result = self._create_final_list(cluster_candidates, k)

        return result
    
    def _create_final_list(self, cluster_candidates, k):
        """
        Create a final list of items from cluster candidates based on the post-processing method.
        """
        if self.post_processing_method == "llm":
            return self._create_llm_list(cluster_candidates, k)
        else: # we have only 2 methods, so if not llm, then round_robin
            return self._create_round_robin_list(cluster_candidates, k)

    def _create_round_robin_list(self, cluster_candidates, k):
        """
        Create a final list of items in a round-robin fashion from cluster candidates.
        """
        final_ids = []
        cluster_ids = list(cluster_candidates.keys())
        i = 0
        while len(final_ids) < k and any(cluster_candidates.values()):
            cluster_id = cluster_ids[i % len(cluster_ids)]
            candidates = cluster_candidates[cluster_id]

            if candidates:
                candidate = candidates.pop(0)
                if candidate not in final_ids:
                    final_ids.append(candidate)
            i += 1

        return final_ids[:k]
    
    def _create_llm_list(self, cluster_candidates, k):
        all_candidates = [item for candidates in cluster_candidates.values() for item in candidates]
        all_candidates_data = []
        for movie_id in all_candidates:
            row = self._loader.items_df.iloc[movie_id]
            all_candidates_data.append((movie_id, row))

        res = self.select_diverse_movies(self.diversity_stimulus, all_candidates_data, k)

        final_ids = [entry.movie_id for entry in res.selected][:k]

        return final_ids

    def select_diverse_movies(self, stimulus , movie_data: list[dict], k: int):
        if stimulus == DiversityStimulus.GENRES:
            return self.select_genre_diverse_movies(movie_data, k)
        elif stimulus == DiversityStimulus.PLOT:
            return self.select_plot_diverse_movies(movie_data, k)
        else: # should not happen... we have only 2 stimuli
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
                - 'movie_id': the ID of the selected movie
                - 'reason': a short explanation for why it was chosen (e.g., unique genres, expands diversity, etc.)

            Eaxmple structure:
            {{
                "selected": [
                    {{"movie_id": 123, "reason": "Your reasoning..."}}
                    ...
                ]
            }}
        """)

        system_prompt = textwrap.dedent("""\
            You are a movie selection assistant helping to maximize diversity in genre across a small list of chosen movies.
            Your job is to pick the most genre-different items from a given list and explain your selection.
            Output must always be in valid JSON format.
        """)

        response = self._run_llama_chat(system_prompt, user_prompt)

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
            - Select {k} distinct movies that are the most diverse from each other based on **plot**.
            - You should aim for the widest thematic and narrative variety, avoiding movies with similar storylines, characters, or settings.

            Return your answer as a JSON object with a key:
            - 'selected': a list of objects, each containing:
                - 'movie_id': the ID of the selected movie
                - 'reason': a short explanation for why it was chosen (e.g., unique story, different narrative style, contrasting themes, etc.)
            
            Example structure:
            {{
                "selected": [
                    {{"movie_id": 123, "reason": "Your reasoning..."}}
                    ...
                ]
            }}
        """)

        system_prompt = textwrap.dedent("""\
            You are a movie selection assistant helping to maximize diversity in movie **plots**.
            Your job is to pick the most plot-different distinct movies from a given list and explain your selection.
            Focus on selecting movies with distinct stories, themes, tones, or narrative styles.
            Output must always be in valid JSON format.
        """)

        response = self._run_llama_chat(system_prompt, user_prompt)

        parsed = SelectionResult.model_validate_json(response.message.content)
        return parsed
    
    def _run_llama_chat(self, system_prompt, user_prompt):
        for _ in range(3):
            try:
                return chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama3.1:8b",
                    format=SelectionResult.model_json_schema(),
                )
            except Exception:
                time.sleep(5)
        raise RuntimeError("Chat failed after 3 attempts")

    def _generate_representant(self, movies_cluster, stimulus: DiversityStimulus):
            
        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_cluster_representant(movies_cluster)
    
    def _generate_diversity_representant(self, representants, stimulus: DiversityStimulus):

        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_diversity_representant(representants)
    
    def _rating_to_directional_effect(self, rating):
        """
        Convert rating to effect:
        - 1 or 2 -> user thinks movies are similar → +delta
        - 4 or 5 -> user thinks movies are diverse → -delta
        - 3 → neutral, no effect
        """
        if rating in (1, 2):
            return 1
        elif rating in (4, 5):
            return -1
        else:
            return 0
        
    def _compute_pair_deltas(self, pairs):
        """
        Compute signed delta for each pair based on plot_sim and genre_sim.
        Positive delta if plot_sim > genre_sim (push toward plot),
        negative if genre_sim > plot_sim (push toward genre).
        """
        deltas = []
        for pair in pairs:
            plot_sim = float(pair['plot_sim'])
            genre_sim = float(pair['genre_sim'])
            diff = abs(plot_sim - genre_sim)
            direction = 1 if plot_sim > genre_sim else -1
            deltas.append(direction * diff)
        return deltas

    def _compute_stimulus(self, pairs, ratings):
        """
        Given pairs and corresponding user ratings,
        compute total stimulus score by summing directional deltas * rating effects,
        then return stimulus type based on the score.
        """
        deltas = self._compute_pair_deltas(pairs)
        stimulus_score = 0.0
        for delta, rating in zip(deltas, ratings):
            effect = self._rating_to_directional_effect(rating)
            stimulus_score += effect * delta

        return DiversityStimulus.GENRES if stimulus_score < 0 else DiversityStimulus.PLOT
    
    def _apply_quantile_transform(self, scores, output_distribution="normal"):
        # Map scores to a specified distribution using quantile transformation
        qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
        return qt.fit_transform(scores)

    @classmethod
    def name(cls):
        return "LLMProfiling"
    
    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "post_processing_method",
                ParameterType.OPTIONS,
                "round_robin",
                options=["round_robin", "llm"],
                help="Choose the post-processing method for generating recommendations.",
                help_key="llm_profiling_post_processing_help",
            )
        ]