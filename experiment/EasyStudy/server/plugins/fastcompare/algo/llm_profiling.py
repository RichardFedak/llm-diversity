from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from ollama import chat
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
)

from plugins.fastcompare.algo.representant_base import (
    Representant,
    DiversityStimulus,
    RepresentantGenerator,
    GenresDiversityHandler,
    PlotDiversityHandler,
)

class LLMProfiling(AlgorithmBase, ABC):

    def __init__(self, loader, **kwargs):
        self._ratings_df = None
        self._loader = None
        self._all_items = None

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
        self._all_items = self._ratings_df.item.unique()

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

        indices = list(selected_items)
        user_vector = np.zeros((items_count,), dtype=np.float32)
        for i in indices:
            user_vector[i] = 1.0

        relevance_scores = np.dot(user_vector, weights)

        rel = np.abs(relevance_scores)
        rel_z = (rel - rel.mean()) / (rel.std() + 1e-8)
        relevance_scores = 1 / (1 + np.exp(-rel_z))

        MAX_CLUSTERS = 3

        ratings = div_perception

        def rating_to_effect(avg_rating):
            return (avg_rating - 3) / 2.0 # rating to [-1, 1] range
        
        def compute_weighted_effect(ratings_data, version_key, sim_key, use_diversity=False):
            items = ratings_data.get(version_key, [])
            if not items:
                return 0.0
            total, weight_sum = 0.0, 0.0
            for item in items:
                rating = int(item["rating"])
                sim = float(item[sim_key])
                effect = rating_to_effect(rating)
                weight = 1.0 - sim if use_diversity else sim
                total += weight * effect
                weight_sum += weight
            return total / weight_sum if weight_sum else 0.0

        plot_effect = compute_weighted_effect(ratings, "no_div_genres", "plot_sim", use_diversity=True) \
                    - compute_weighted_effect(ratings, "no_div_plot", "plot_sim", use_diversity=False)

        genre_effect = compute_weighted_effect(ratings, "no_div_plot", "genre_sim", use_diversity=True) \
                    - compute_weighted_effect(ratings, "no_div_genres", "genre_sim", use_diversity=False)

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
        mask[filter_out_items] = 0
        relevance_scores[filter_out_items] = 0
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

            cluster_mask = ~np.isin(cluster_labels, top_clusters)
            #print(cluster_labels)
            cluster_labels[cluster_mask] = -1
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
            # multiply similarities by preds item wise
            similarities = similarities * relevance_scores[mask]
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

        #print("LLMprofiling done")

        return result[:k]

    def _generate_representant(self, movies_cluster, stimulus: DiversityStimulus):
            
        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_cluster_representant(movies_cluster)
    
    def _generate_diversity_representant(self, representants, stimulus: DiversityStimulus):

        handler = self.stimulus_handlers[stimulus]()

        return handler.generate_diversity_representant(representants)

    @classmethod
    def name(cls):
        return "LLMProfiling"
    
    @classmethod
    def parameters(cls):
        return []