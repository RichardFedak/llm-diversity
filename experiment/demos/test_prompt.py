from ollama import chat
from pydantic import BaseModel
from typing import List

class Movie(BaseModel):
    title: str

class MovieList(BaseModel):
    movies: List[Movie]

favorite_movies = [
    Movie(title="Harry Potter and the Sorcerer's Stone"),
    Movie(title="Harry Potter and the Chamber of Secrets"),
    Movie(title="Harry Potter and the Prisoner of Azkaban"),
    Movie(title="Interstellar"),
    Movie(title="The Matrix"),
    Movie(title="Inception")
]

system_prompt = f"""
You are a helpful assistant recommending movies.
Here are movies the user likes:

""" + "\n".join([f"- {movie.title}" for movie in favorite_movies])

response = chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Recommend 5 movies."}
    ],
    model="llama3.2",
    format=MovieList.model_json_schema(),
)

movie_list = MovieList.model_validate_json(response.message.content)
for movie in movie_list.movies:
    print(movie.title)
