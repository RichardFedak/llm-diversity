from abc import ABC, abstractmethod
from enum import Enum
import textwrap
from ollama import chat
from pydantic import BaseModel
import time

class Representant(BaseModel):
    genres: str
    plot: str

class DiversityStimulus(str, Enum):
    GENRES = "genres"
    PLOT = "plot"

def get_stimulus_weight(stimulus: DiversityStimulus) -> float:
    if stimulus == DiversityStimulus.GENRES:
        return 0.6
    elif stimulus == DiversityStimulus.PLOT:
        return 0.8

class RepresentantGenerator(ABC):
    def __init__(self, chat_model="llama3.1:8b"):
        self.chat_model = chat_model

    @abstractmethod
    def generate_cluster_representant(self, movies_cluster) -> 'Representant':
        pass

    @abstractmethod
    def generate_diversity_representant(self, representants) -> 'Representant':
        pass

class GenresDiversityHandler(RepresentantGenerator):

    def _call_chat(self, system_prompt: str, user_prompt: str) -> 'Representant':
        for _ in range(3):
            try:
                r = chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.chat_model,
                    format=Representant.model_json_schema(),
                )
                return Representant.model_validate_json(r.message.content)
            except Exception:
                time.sleep(5)
        raise RuntimeError("Chat failed after 3 attempts")

    def generate_cluster_representant(self, movies_cluster):
        movie_text = "\n".join(
            f"- {m['title']} | Genres: {m['genres']} Plot: {m['plot']}"
            for m in movies_cluster
        )

        user_prompt = textwrap.dedent(f"""\
            The user enjoys the following movies:

            {movie_text}

            Based on these, write a new synthetic movie-style entry.

            Your task:
                - Identify the common genres shared across the listed movies.
                - Create a new plot that fits and reflects those shared genres.

            Return a JSON object with:
                - 'genres': a comma-separated list of the most commonly shared genres
                - 'plot': a short, original movie-style description that fits those genres
        """)

        system_prompt = textwrap.dedent("""\
            You are a creative assistant that synthesizes a user’s taste in movies by analyzing their shared genre patterns.

            Your task:
                - Detect the most commonly shared genres across the provided movies
                - Generate a believable and engaging new plot that fits those genres
                - Ensure the result reflects the user's preferences

            Return a JSON object with:
                - 'genres': a comma-separated list of the most commonly shared genres
                - 'plot': a short, original movie-style description that fits those genres
        """)

        return self._call_chat(system_prompt, user_prompt)

    def generate_diversity_representant(self, representants):
        reps = "\n".join(
            f"- Genres: {r.genres} | Plot: {r.plot}"
            for r in representants
        )
        user_prompt = textwrap.dedent(f"""\    
            Below are several representants, each summarizing a group of movies the user enjoys:

            {reps}

            Your task:
                - Identify genres the user already enjoys.
                - Suggest new genres that introduce diversity but remain plausible and appealing to the user’s taste (avoid genres that would feel out of character).
                - Create a synthetic 'representant' with those new genres.
                - Come up with a plot that will fit the new genres.

            Return a JSON object with:
                - 'genres': a comma-separated string of the new genres
                - 'plot': a short movie-style generated description that fits the new genres
        """)

        system_prompt = textwrap.dedent("""\
            You are a creative assistant generating diversity-focused movie profiles.
            Given several 'representants' that summarize the user's known preferences,
            your task is to expand their profile by introducing new, diverse genres that the user would likely enjoy,
            avoiding genres that would feel unrelated or inconsistent with their taste.

            Your task:
                - Identify existing genres in the input.
                - Suggest new genres that diversify but still suit the user's preferences.
                - Generate a plot description that matches the new genres.

            Return a JSON object with:
                - 'genres': a comma-separated string of the new genres
                - 'plot': a short movie-style generated description that fits the new genres
        """)

        return self._call_chat(system_prompt, user_prompt)

class PlotDiversityHandler(RepresentantGenerator):

    def _call_chat(self, system_prompt: str, user_prompt: str) -> 'Representant':
        for _ in range(3):
            try:
                r = chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.chat_model,
                    format=Representant.model_json_schema(),
                )
                return Representant.model_validate_json(r.message.content)
            except Exception:
                time.sleep(5)
        raise RuntimeError("Chat failed after 3 attempts")

    def generate_cluster_representant(self, movies_cluster):
        # similar to GenresHandler, but with emphasis on plot synthesis
        movie_text = "\n".join(
            f"- {m['title']} | Genres: {m['genres']} Plot: {m['plot']}"
            for m in movies_cluster
        )
        user_prompt = textwrap.dedent(f"""\
            The user enjoys the following movies:

            {movie_text}

            Based on these, write a new synthetic movie-style entry.
            
            Your task:
                - Create a new plot that reflects the storytelling patterns and themes of the listed movies.
                - Then, assign genres that best describe this new plot.

            Return a JSON object with:
                - 'plot': a short, original movie-style description inspired by the plots of the listed movies
                - 'genres': a comma-separated list of genres that best fit the new plot
        """)
        system_prompt = textwrap.dedent("""\
            You are a creative assistant that synthesizes a user’s taste in movies by analyzing a group of movies.
            
            Your task:
                - Understand the common themes, tone, and narrative structure of the provided movies.
                - Create an original plot that reflects those shared qualities.
                - Assign fitting genres based on the newly written plot.

            Return a JSON object with:
                - 'plot': a short, original movie-style description inspired by the plots of the listed movies
                - 'genres': a comma-separated list of genres that best fit the new plot
        """)

        return self._call_chat(system_prompt, user_prompt)

    def generate_diversity_representant(self, representants):
        reps = "\n".join(
            f"- Genres: {r.genres} | Plot: {r.plot}"
            for r in representants
        )
        user_prompt = textwrap.dedent(f"""\  
            Below are several representants, each summarizing a group of movies the user enjoys:

            {reps}

            Your task:
                - Create a **new and original plot** that is different from any of the plots of representants.
                - Ensure the new plot is plausible and aligns with the user’s preferences, avoiding themes or patterns that would feel out of character.
                - Based on the new plot, infer and list the genres that best describe it.

            Return a JSON object with:
                - 'plot': a short, movie-style generated description that avoids similarities with the input plots but respects the user's taste.
                - 'genres': a comma-separated string of genres that best match the generated plot.
        """)

        system_prompt = textwrap.dedent("""\
            You are a creative assistant generating diversity-focused movie profiles.
            Given several 'representants' summarizing the user's known movie preferences,
            your task is to expand their profile by generating a new plot.
                                        
            Your task:
                - Come up wth a plot that is original and distinct from all existing plots.
                - The plot reflects the user's probable tastes, avoiding completely unrelated themes.
                - After creating the plot, assign genres that best describe it.

            Return a JSON object with:
                - 'plot': a short, movie-style generated description that avoids similarities with the input plots but respects the user's taste.
                - 'genres': a comma-separated string of genres that best match the generated plot.
        """)

        return self._call_chat(system_prompt, user_prompt)

