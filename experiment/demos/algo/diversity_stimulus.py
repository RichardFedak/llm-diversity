import json
import math
from enum import Enum
from typing import List, Dict
from ollama_wrapper import OllamaWrapper
from pydantic import BaseModel
from model import MovieItem


class StimulusType(Enum):
    GENRES = "genres"
    PLOT = "plot"
    # Future stimuli can go here...

class StimulusResponseRecommend(BaseModel):
    list_analysis: str
    list_score: int

class StimulusResponseAnalyze(BaseModel):
    list_1_analysis: str
    list_1_score: int
    list_2_analysis: str
    list_2_score: int
    list_3_analysis: str
    list_3_score: int


class DiversityStimulusAnalyzer:
    def __init__(self):
        self.stimuli_handlers = {
            stimulus: (lambda stim: lambda options, perceived_index = None, weights = None: self._analyze_stimulus(stim, options, perceived_index, weights))(stimulus)
            for stimulus in StimulusType
        }
        self.log_path = "stimulus_logs.json"
        self.stimulus_totals: Dict[StimulusType, int] = {stimulus: 0 for stimulus in StimulusType}
        self.analysis_count: int = 0

    def analyze(
        self,
        options: List[List[MovieItem]],
        perceived_most_diverse_index: int = None,  # Used for metric assessment
        weights: List[int] = None                  # Used for magnitude assessment
    ) -> Dict[StimulusType, float]:

        analysis_results = {}

        for stimulus, handler in self.stimuli_handlers.items():
            diversity_score = handler(options, perceived_most_diverse_index, weights)
            analysis_results[stimulus] = diversity_score
            self.stimulus_totals[stimulus] += diversity_score

        return self._normalize_results(analysis_results)

    def _log_to_json(self, stimulus: StimulusType, prompt: str, response, diversity_score: int):
        log_entry = {
            "stimulus_type": stimulus.value,
            "prompt": prompt,
            "response": response.model_dump(),
            "diversity_score": diversity_score,
        }

        try:
            with open(self.log_path, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        except FileNotFoundError:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=2)

    def _analyze_stimulus(
        self,
        stimulus: StimulusType,
        options: List[List[MovieItem]],
        perceived_index: int,
        weights: List[float]
    ) -> int:
        field_name = stimulus.value  # "genres" or "plot"

        system_prompt = (
            f"You are an expert in analyzing movie collections for diversity based solely on {field_name.upper()}. "
            f"You will receive 3 movie lists. Each movie includes a title and its {field_name}. "
            f"Your task is to analyze each list individually and determine how diverse it is in terms of {field_name} of its movies."
        )

        STIMULUS_INSTRUCTIONS = {
            StimulusType.GENRES: (
                "Focus only on the listed genres.\n"
                "Consider genre variety and uniqueness across movies in each list.\n"
                "Avoid interpreting the genre meaning — just evaluate the diversity based on genre labels."
            ),
            StimulusType.PLOT: (
                "Focus only on the content of the plots. Do not focus on genres.\n"
                "Try to group plots into themes and assess the thematic range in each list.\n"
                "Summarize the dominant themes of the list in your analysis."
            )
        }


        prompt = (
            f"Please do the following:\n"
            f"1. For each list, provide a short analysis of its {field_name} diversity.\n"
            f"{STIMULUS_INSTRUCTIONS[stimulus]}\n"
            f"2. Give a diversity score between 0 and 10 for each list (10 = very diverse, 0 = no diversity in terms of {field_name}).\n"
        )

        for i, movie_list in enumerate(options):
            prompt += f"List {i + 1}:\n"
            for movie in movie_list:
                value = getattr(movie, field_name)
                prompt += f"- {movie.title} | {field_name.capitalize()}: {value}\n"
            prompt += "\n"

        prompt += (
            "Format your response as JSON with these keys:\n"
            "- per_list_analysis: list of strings (one per list)\n"
            "- per_list_scores: list of integers (one per list)\n"
        )

        wrapper = OllamaWrapper[StimulusResponseAnalyze](
            system_prompt=system_prompt,
            response_model=StimulusResponseAnalyze,
        )

        response = wrapper.ask(prompt)

        diversity_score = None

        if weights is not None:
            # Multiply diversity score by the weight (approx alpha)
            transformed_weights = self._transform_weights_with_sigmoid(weights)
            diversity_score = sum(
                getattr(response, f"list_{i + 1}_score") * transformed_weights[i]
                for i in range(len(weights))
            )
        elif perceived_index is not None:
            diversity_score = getattr(response, f"list_{perceived_index}_score")
        else:
            raise ValueError("Either perceived_most_diverse_index or weights must be provided.")

        self._log_to_json(stimulus, prompt, response, diversity_score)

        return diversity_score

    def _transform_weights_with_sigmoid(self, weights: List[float], sharpness: float = 10.0) -> List[float]:
        def sigmoid(x):
            return 2 / (1 + math.exp(-sharpness * (x - 0.5)))
        return [sigmoid(w) for w in weights]

    def _normalize_results(
        self,
        raw_results: Dict[StimulusType, int]
    ) -> Dict[StimulusType, float]:
        total = sum(raw_results.values())
        if total == 0:
            return {stimulus: 0.0 for stimulus in raw_results}
        return {
            stimulus: score / total
            for stimulus, score in raw_results.items()
        }
    
    def calculate_total_distribution(self, stretch_factor: float = 0.5) -> Dict[StimulusType, float]:
        total = sum(self.stimulus_totals.values())
        if total == 0:
            return {stimulus: 0.0 for stimulus in StimulusType}

        normalized = {
            stimulus: score / total
            for stimulus, score in self.stimulus_totals.items()
        }

        stretched = {
            stimulus: val ** (1 / stretch_factor)
            for stimulus, val in normalized.items()
        }

        stretched_total = sum(stretched.values())
        return {
            stimulus: val / stretched_total
            for stimulus, val in stretched.items()
        }


class DiversityStimulusEvaluator:
    def __init__(self, stimuli_scores: Dict[StimulusType, float]):
        self.stimuli_scores = stimuli_scores

    def _analyze_diversity_for_list(self, stimulus: StimulusType, movie_list: List[MovieItem]) -> int:

        field_name = stimulus.value  # "genres" or "plot"
        
        system_prompt = (
            f"You are an expert in analyzing movie collections for diversity based solely on {field_name.upper()}. "
            f"You will receive one movie list. Each movie includes a title and its {field_name}. "
            f"Your task is to analyze the list and determine how diverse it is in terms of {field_name}."
        )

        STIMULUS_INSTRUCTIONS = {
            StimulusType.GENRES: (
                "Focus only on the listed genres.\n"
                "Consider genre variety and uniqueness across movies in the list.\n"
                "Avoid interpreting the genre meaning — just evaluate the diversity based on genre labels."
            ),
            StimulusType.PLOT: (
                "Focus only on the content of the plots. Do not focus on genres.\n"
                "Try to group plots into themes and assess the thematic range in the list.\n"
                "Summarize the dominant themes of the list in your analysis."
            )
        }

        prompt = (
            f"Please do the following:\n"
            f"1. Provide a short analysis of the {field_name} diversity for the list.\n"
            f"{STIMULUS_INSTRUCTIONS[stimulus]}\n"
            f"2. Give a diversity score between 0 and 10 for the list (10 = very diverse, 0 = no diversity in terms of {field_name}).\n\n"
        )

        for movie in movie_list:
            value = getattr(movie, field_name)
            prompt += f"- {movie.title} | {field_name.capitalize()}: {value}\n"
        
        prompt += (
            "Format your response as JSON with these keys:\n"
            "- list_analysis: string\n"
            "- list_score: integer\n"
        )

        wrapper = OllamaWrapper[StimulusResponseRecommend](
            system_prompt=system_prompt,
            response_model=StimulusResponseRecommend,
        )

        response = wrapper.ask(prompt)

        return response.list_score

    def evaluate_lists(self, movie_lists: List[List[MovieItem]]) -> float:
        total_score = 0
        total_weight = 0

        for movie_list in movie_lists:
            for stimulus in StimulusType:
                list_score = self._analyze_diversity_for_list(stimulus, movie_list)
                weight = self.stimuli_scores[stimulus]
                total_score += list_score * weight
                total_weight += weight

        final_score = total_score / total_weight
        return final_score
