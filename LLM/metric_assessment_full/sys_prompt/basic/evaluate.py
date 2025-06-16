import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
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

with open("final_movie_data_with_summary.json", 'r') as f:
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
You are an assistant tasked with comparing three lists of movies. For each movie, you are given its: {field_text}
Use your own judgment to determine what information is relevant when assessing the diversity of the lists. 
You should consider the given {field_text} for every movie, as well as an overall summary of the entire list.

The summary includes the following information:
    - Popularity Diversity: A score based on the mix of blockbuster movies and niche/independent films. The higher the score, the more balanced and more diverse is the list in this category.
    - Genre Diversity: A score based on the variety of genres represented in the list. The higher the score, the more diverse the genres.
    - Theme Diversity: A score based on the thematic range of the movies in the list. The higher the score, the more varied and expansive the themes across the movies in the list.
    - Time Span: Years of the earliest and latest movie releases in the list.
    - Franchise Inclusion: Whether the list includes at least two movies from the same franchise.

Deliver your descriptions of lists, comparison, and choice of the most diverse list in the following JSON format:

{{
    "list_A_description": string,            # Describe the diversity of the movies in list A
    "list_B_description": string,            # Describe the diversity of the movies in list B
    "list_C_description": string,            # Describe the diversity of the movies in list C
    "comparison": string,                    # Compare the diversity of the lists.
    "most_diverse_list_reasoning": string,   # Explanation of which list you perceive to be the most diverse.
    "most_diverse_list": string              # The list you determine to be the most diverse, either 'A', 'B', or 'C'.
}}
"""

for fields in field_combinations:
    system_prompt = generate_prompt(fields)
    evaluation_name = "_summary" + "_".join(field.name.lower() for field in fields)

    print(f"Running evaluation for: {evaluation_name}")

    evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
    evaluator.evaluate_data(data)
    
    print(f"Completed evaluation for: {evaluation_name}\n")
