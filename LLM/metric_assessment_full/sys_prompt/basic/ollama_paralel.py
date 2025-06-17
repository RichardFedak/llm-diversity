import time
from pydantic import BaseModel
from ollama import chat
from concurrent.futures import ThreadPoolExecutor

# System prompt to enforce JSON format
system_prompt = """
You are a helpful assistant that provides information about movies.
Always provide the answer in the following JSON format:
{
    "movie_description": string,
    "release_year": int
}
"""

# Pydantic model for validation
class Response(BaseModel):
    movie_description: str
    release_year: int

# Prompts to be queried
movie_prompts = [
    "Please, tell me something about the movie 'Inception'.",
    "Please, tell me something about the movie 'Titanic'.",
    "Please, tell me something about the movie 'The Matrix'."
]

# Function to query the model
def query_movie(prompt: str) -> Response:
    raw_response = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model="llama3.1:8b",
        format=Response.model_json_schema(),
    )
    return Response.model_validate_json(raw_response.message.content)

# Sequential execution
print("== Sequential Execution ==")
start_seq = time.time()
sequential_results = [query_movie(prompt) for prompt in movie_prompts]
end_seq = time.time()
print("== Sequential End ==")

# Parallel execution
print("== Parallel Execution ==")
start_par = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    parallel_results = list(executor.map(query_movie, movie_prompts))
end_par = time.time()
print("== Parallel End ==")

# Output results
print("== Sequential Results ==")
for res in sequential_results:
    print(f"{res.movie_description} (Released: {res.release_year})")
print(f"Sequential Time: {end_seq - start_seq:.2f} seconds\n")

print("== Parallel Results ==")
for res in parallel_results:
    print(f"{res.movie_description} (Released: {res.release_year})")
print(f"Parallel Time: {end_par - start_par:.2f} seconds")
