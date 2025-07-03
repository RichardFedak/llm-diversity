from pydantic import BaseModel
from ollama import chat

system_prompt = """
You are a helpful assistant that provides information about movies.
Always provide the answer in the following JSON format:
{
    "movie_description": string,   # A short description of the movie
    "release_year": int             # The year the movie was released
}
"""

class Response(BaseModel):
    movie_desc: str
    year: int

prompt = "Please, tell me something about the movie 'Avatar'."

raw_response = chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    model="llama3.1:8b",
    format=Response.model_json_schema(),
)

data = Response.model_validate_json(raw_response.message.content)
print(data.movie_desc)  # 'Avatar' is a 2009 American epic science ...
print(data.year)        # 2009