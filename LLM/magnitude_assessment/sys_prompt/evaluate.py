import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module.evaluator import MovieEvaluator, MovieFields

api_key = os.getenv("GEMINI_API_KEY")
system_prompt = """
You are an assistant tasked with comparing three lists of movies. For each movie, the title is provided. 
Use your own judgment to determine what information is relevant when assessing the diversity of the lists. You may consider the given titles.

Deliver your descriptions of lists, comparison and choice of the most diverse list in the following JSON format:

{
    "list_A_description: string               # Describe the diversity of the movies in the list A
    "list_B_description: string               # Describe the diversity of the movies in the list B
    "list_C_description: string               # Describe the diversity of the movies in the list C
    "comparison": string,                     # Compare the diversity of the lists.
    "diversity_scores": dict                   # A dictionary assigning a diversity score (0-10) to each list based on its relative diversity. Example: {"A": 5, "B": 2, "C": 10} (where C is the most diverse and B the least).
}
"""

input_fields = [MovieFields.TITLE]
evaluation_name = "title"

with open("final_movie_data_with_summary.json", 'r') as f:
    data = json.load(f)

evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, input_fields)
evaluator.evaluate_data(data)