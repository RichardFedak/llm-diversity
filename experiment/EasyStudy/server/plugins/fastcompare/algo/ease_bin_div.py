from abc import ABC

import functools
import numpy as np
import pandas as pd
import scipy
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

from sklearn.preprocessing import QuantileTransformer


class Binomial_diversity:
        def __init__(self, all_categories, get_item_categories, p_g_dict):
            self.all_categories = set(all_categories)
            self.category_getter = get_item_categories
            self._p_g_dict = p_g_dict

        def __call__(self, rec_list):
            rec_list_categories = self._get_list_categories(rec_list)
            return self._coverage(rec_list, rec_list_categories) * self._non_red(rec_list, rec_list_categories)

        @functools.lru_cache(maxsize=10000)
        def _n_choose_k(self, N, k):
            return scipy.special.comb(N, k)

        # Calculate for each genre separately
        # N is length of the recommendation list
        # k is number of successes, that is, number of items belonging to that genre
        # For each genre, recommendation list is sequence of bernouli trials, and each item in the list having the genre is considered to be a success
        # Calculate probability of k successes in N trials
        def _binomial_probability(self, N, k, p):
            return self._n_choose_k(N, k) * np.power(p, k) * np.power(1.0 - p, N - k) 

        def _get_list_categories(self, rec_list):
            categories = []
            for item in rec_list:
                categories.extend(self.category_getter(item))
            return set(categories)

        # Coverage as in the Binomial algorithm paper
        def _coverage(self, rec_list, rec_list_categories):
            #rec_list_categories = self._get_list_categories(rec_list)
            not_rec_list_categories = self.all_categories - rec_list_categories

            N = len(rec_list)
            
            prod = 1
            for g in not_rec_list_categories:
                p = self._p_g_dict[g]
                prod *= np.power(self._binomial_probability(N, 0, p), 1.0 / len(self.all_categories))            
            return prod
        
        # Corresponds to conditional probability used in formulas (10) and (11) in original paper
        def _category_redundancy(self, g, k_g, N):
            s = 0.0
            for l in range(1, k_g):
                # We want P(x_g = l | X_g > 0) so rewrite it as P(x_g = l & X_g > 0) / P(X_g > 0)
                # P(x_g = l & X_g > 0) happens when P(x_g = l) is it already imply X_g > 0
                # so we further simplify this as P(x_g = l) / P(X_g > 0) and P(X_g > 0) can be set to 1 - P(X_g = 0)
                # so we end up with
                # P(x_g = l) / (1 - P(X_g = 0))
                p = self._p_g_dict[g]
                s += (self._binomial_probability(N, l, p) / (1.0 - self._binomial_probability(N, 0, p)))

            return np.clip(1.0 - s, 0.0, 1.0)
        
        def _non_red(self, rec_list, rec_list_categories):
            #rec_list_categories = self._get_list_categories(rec_list)

            N = len(rec_list)
            N_LIST_CATEGORIES = len(rec_list_categories)

            prod = 1.0
            for g in rec_list_categories:
                #num_movies_with_genre = get_num_movies_with_genre(rec_list, g)
                k_g = len([x for x in rec_list if g in self.category_getter(x)])
                p_cond = self._category_redundancy(g, k_g, N)
                prod *= np.power(p_cond, 1.0 / N_LIST_CATEGORIES)

            return prod
        
        # Corresponds to conditional probability used in formulas (10) and (11) in original paper
        def _category_redundancy(self, g, k_g, N):
            s = 0.0
            for l in range(1, k_g):
                # We want P(x_g = l | X_g > 0) so rewrite it as P(x_g = l & X_g > 0) / P(X_g > 0)
                # P(x_g = l & X_g > 0) happens when P(x_g = l) is it already imply X_g > 0
                # so we further simplify this as P(x_g = l) / P(X_g > 0) and P(X_g > 0) can be set to 1 - P(X_g = 0)
                # so we end up with
                # P(x_g = l) / (1 - P(X_g = 0))
                p = self._p_g_dict[g]
                s += (self._binomial_probability(N, l, p) / (1.0 - self._binomial_probability(N, 0, p)))

            return np.clip(1.0 - s, 0.0, 1.0)

class EASE_BinDiv(AlgorithmBase, ABC):
    """Implementation of EASE algorithm + postprocessing utilizing diversification
    procedure with binomial diversity metric
    paper: https://dl.acm.org/doi/10.1145/3627043.3659555
    """

    def __init__(self, loader, alpha, n_candidates, positive_threshold, l2, **kwargs):
        self._ratings_df = None
        self._loader = loader
        self._all_items = None

        self._item_data = None
        self._all_categories = None
        self._get_item_idx_categories = None
        
        self._rating_matrix = None
        self._p_g_dict = None
        self._diversity_function = None

        self._threshold = positive_threshold
        self._l2 = l2
        self._n_candidates = n_candidates
        self._alpha = alpha
        self.NEG_INF = int(-10e6)

    # One-time fitting of the algorithm for a predefined number of iterations
    def fit(self, loader):

        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()

        self._item_data = loader.items_df
        self._all_categories = loader.get_all_categories()
        self._get_item_idx_categories = loader.get_item_index_categories

        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )

        x = self._rating_matrix.astype(bool)
        denom = x.sum()
        item_counts = x.sum(axis=0)
        p_g_dict = {g: 0.0 for g in self._all_categories}

        for item, cnt in enumerate(item_counts):
            if cnt > 0:
                for g in self._get_item_idx_categories(item):
                    p_g_dict[g] += cnt

        for g in p_g_dict:
            p_g_dict[g] /= denom

        self._p_g_dict = p_g_dict

        self._diversity_function = Binomial_diversity(
            self._all_categories,
            self._get_item_idx_categories,
            self._p_g_dict
            )

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k, weights, items_count, div_perception):
        #print("PREDICTING EASE BIN DIV")
        rat = pd.DataFrame({"item": selected_items}).set_index("item", drop=False)
        # Appropriately filter out what was seen and what else should be filtered
        candidates = np.setdiff1d(self._all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)
        if not selected_items:
            # Nothing was selected, since the new_user was unknown during training, Lenskit algorithm would simply recommended nothing
            # to avoid empty recommendation, we just sample random candidates
            return np.random.choice(candidates, size=k, replace=False).tolist()
        indices = list(selected_items)
        user_vector = np.zeros((items_count,), dtype=np.float32)

        user_vector[indices] = 1

        rel_scores = np.dot(user_vector, weights)

        #print("rel scores", rel_scores.shape)
        #print("filter out", filter_out_items)

        # mask out scores for already seen movies
        rel_scores[selected_items] = self.NEG_INF
        rel_scores[filter_out_items] = self.NEG_INF

        result = self.diversify(
            k=k, 
            rel_scores=rel_scores,
            alpha=self._alpha,
            diversity_f=self._diversity_function,
            rating_row=user_vector,
            filter_out_items=filter_out_items
            )
        
        #print("EASE BIN DIV PREDICTION DONE")

        return result

    def diversify(self, k, rel_scores,
        alpha, diversity_f,
        rating_row, filter_out_items):

        assert rel_scores.ndim == 1
        assert rating_row.ndim == 1

        def relevance_f(top_k):
            return rel_scores[top_k].sum()

        n_items_subset= self._n_candidates

        # This is going to be the resulting top-k item
        top_k_list = np.zeros(shape=(k, ), dtype=np.int32)

        # Hold marginal gain for each item, objective pair
        mgains = np.zeros(shape=(2, n_items_subset), dtype=np.float32)

        # Sort relevances
        # Filter_out_items are already propageted into rel_scores (have lowest score)
        sorted_relevances = np.argsort(-rel_scores, axis=-1)

        assert n_items_subset % 2 == 0, f"When using random mixture we expect n_items_subset ({n_items_subset}) to be divisible by 2"
        # Here we need to ensure that we do not include already seen items among source_items
        # so we have to filter out 'filter_out_items' out of the set

        # We know items from filter_out_items have very low relevances
        # so here we are safe w.r.t. filter_out_movies because those will be at the end of the sorted list
        relevance_half = sorted_relevances[:n_items_subset//2]
        # However, for the random half, we have to ensure we do not sample movies from filter_out_movies because this can lead to issues
        # especially when n_items_subset is small and filter_out_items is large (worst case is that we will sample exactly those items that should have been filtered out)
        random_candidates = np.setdiff1d(sorted_relevances[n_items_subset//2:], filter_out_items)
        random_half = np.random.choice(random_candidates, n_items_subset//2, replace=False)

        source_items = np.concatenate([
            relevance_half, 
            random_half
        ])

        # Default number of quantiles is 1000, however, if n_samples is smaller than n_quantiles, then n_samples is used and warning is raised
        # to get rid of the warning, we calculates quantiles straight away
        n_quantiles = min(1000, mgains.shape[1])

        # Mask-out seen items by multiplying with zero
        # i.e. 1 is unseen
        # 0 is seen
        # Lets first set zeros everywhere
        seen_items_mask = np.zeros(shape=(source_items.size, ), dtype=np.int32)
        # And only put 1 to UNSEEN items in CANDIDATE (source_items) list
        seen_items_mask[rating_row[source_items] <= 0.0] = 1
        
        # Build the recommendation incrementally
        for i in range(k):
            # Relevance and diversity
            for obj_idx, obj_func in enumerate([relevance_f, diversity_f]):
                # Cache f_prev
                f_prev = obj_func(top_k_list[:i])
                
                objective_cdf_train_data = []
                # For every source item, try to add it and calculate its marginal gain
                for j, item in enumerate(source_items):
                    top_k_list[i] = item # try extending the list
                    objective_cdf_train_data.append(obj_func(top_k_list[:i+1]) - f_prev)
                    mgains[obj_idx, j] = objective_cdf_train_data[-1]
                    
                # Use cdf_div to normalize marginal gains
                mgains[obj_idx] = QuantileTransformer(n_quantiles=n_quantiles).fit_transform(mgains[obj_idx].reshape(-1, 1)).reshape(mgains[obj_idx].shape)
        
            # Select next best item to be added (incrementally) to the recommendation list
            best_item_idx = self.select_next(alpha, mgains, seen_items_mask)
            best_item = source_items[best_item_idx]
                
            # Select the best item and append it to the recommendation list            
            top_k_list[i] = best_item
            # Mask out the item so that we do not recommend it again
            seen_items_mask[best_item_idx] = 0

        return top_k_list



    def mask_scores(self, scores, seen_items_mask):
        
        # Ensure seen items get lowest score of 0
        # Just multiplying by zero does not work when scores are not normalized to be always positive
        # because masked-out items will not have smallest score (some valid, non-masked ones can be negative)
        # scores = scores * seen_items_mask[user_idx]
        # So instead we do scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])
        min_score = scores.min()
        # Here we do not mandate NEG_INF to be strictly smaller
        # because rel_scores may already contain some NEG_INF that was set by predict_with_score
        # called previously -> so we allow <=.
        scores = scores * seen_items_mask + self.NEG_INF * (1 - seen_items_mask)
        return scores


    # Selects next candidate with highest score
    # Calculate scores as (1 - alpha) * mgain_rel + alpha * mgain_div
    # that is used in diversification experiments
    def select_next(self, alpha, mgains, seen_items_mask):
        # We ignore seen items mask here
        assert mgains.ndim == 2 and mgains.shape[0] == 2, f"shape={mgains.shape}"
        scores = (1.0 - alpha) * mgains[0] + alpha * mgains[1]
        assert scores.ndim == 1 and scores.shape[0] == mgains.shape[1], f"shape={scores.shape}"
        scores = self.mask_scores(scores, seen_items_mask)
        return scores.argmax()

    @classmethod
    def name(cls):
        return "EASE + BinDiv"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "l2",
                ParameterType.FLOAT,
                500,  # I did not find a value in the paper, we can try tweaking the default value in the future
                help="L2-norm regularization",
                help_key="ease_l2_help",
            ),
            Parameter(  # at the moment, we assume that greater ratings are better
                "positive_threshold",
                ParameterType.FLOAT,
                2.5,
                help="Threshold for conversion of n-ary rating into binary (positive/negative).",
            ),
            Parameter(
                "alpha",
                ParameterType.FLOAT,
                0.5,
                help="Diversification strength, higher alpha means higher diversification.",
                help_key="ease_alpha_help",
            ),
            Parameter(
                "n_candidates",
                ParameterType.INT,
                100,
                help="How many candidate items should we consider for diversification procedure. Should be divisible by 2.",
            ),
        ]
