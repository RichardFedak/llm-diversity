import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module.evaluator import MovieEvaluator, MovieFields

api_key = os.getenv("GEMINI_API_KEY")

field_combinations = [
    [MovieFields.TITLE],
    [MovieFields.TITLE, MovieFields.GENRES],
    [MovieFields.TITLE, MovieFields.PLOT],
    [MovieFields.TITLE, MovieFields.GENRES, MovieFields.PLOT],
    [MovieFields.GENRES],
    [MovieFields.PLOT],
    [MovieFields.GENRES, MovieFields.PLOT]
]

with open("final_movie_data.json", 'r') as f:
    data = json.load(f)

def generate_prompt(fields):
    field_descriptions = []
    
    if MovieFields.TITLE in fields:
        field_descriptions.append("title")
    if MovieFields.GENRES in fields:
        field_descriptions.append("genres")
    if MovieFields.PLOT in fields:
        field_descriptions.append("plot")
    
    field_text = ", ".join(field_descriptions)
    
    return f"""
You are an assistant tasked to assess diversity of 6 lists of movies. For each movie, you are given its: {field_text}
The main goal is to assign an overall diversity score to the lists in the range of -3 to 3, with -3 representing no diversity and 3 representing high diversity.
You should consider the given {field_text} for every movie.

Deliver your descriptions of lists, overall diversity summarization, and diversity score for the entire batch, in the following JSON format:

{{
    "list_A_description": string,            # Describe the diversity of the movies in list A.
    "list_B_description": string,            # Describe the diversity of the movies in list B.
    "list_C_description": string,            # Describe the diversity of the movies in list C.
    "list_D_description": string,            # Describe the diversity of the movies in list D.
    "list_E_description": string,            # Describe the diversity of the movies in list E.
    "list_F_description": string,            # Describe the diversity of the movies in list F.
    "diversity_summarization": string,       # Summarize the diversity across 6 provided lists.
    "diversity_score": int                   # Assign a diversity score between -3 and 3, excluding 0, with -3 representing no diversity and 3 for high diversity of movies across the lists.
}}
"""

for fields in field_combinations:
    system_prompt = generate_prompt(fields)
    evaluation_name = "_".join(field.name.lower() for field in fields)

    print(f"Running evaluation for: {evaluation_name}")

    evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
    evaluator.evaluate_data(data)
    
    print(f"Completed evaluation for: {evaluation_name}\n")
