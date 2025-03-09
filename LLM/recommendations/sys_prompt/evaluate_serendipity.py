import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module.evaluator_serendipity import MovieEvaluator, MovieFields

api_key = os.getenv("GEMINI_API_KEY")

field_combinations = [
    [MovieFields.TITLE],
    [MovieFields.TITLE, MovieFields.GENRES],
    [MovieFields.TITLE, MovieFields.PLOT],
    [MovieFields.TITLE, MovieFields.GENRES, MovieFields.PLOT],
    # [MovieFields.GENRES],
    # [MovieFields.PLOT],
    # [MovieFields.GENRES, MovieFields.PLOT]
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
You are an assistant tasked with assessing serendipity (how unexpected yet interesting the movies are) across 6 lists. 
For each movie, you are given its: {field_text}.

Your task is to analyze each list’s selections and identify movies that contribute to serendipity—those that were chosen but stand out from others due to unique features, themes, or characteristics.

After analyzing all lists:
1. Summarize the overall serendipity across the lists by answering: Were the movies in each list highly unexpected yet interesting?
2. Answer the question: Were the movies across the lists unexpected yet interesting?
3. Based on a 6-point Likert scale (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3), choose the most appropriate option for the question.

Deliver your analysis in the following JSON format:

{{
    "list_A_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "list_B_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "list_C_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "list_D_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "list_E_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "list_F_analysis": string,               # Explain possible reasons why the user selected the movies and if any of them have some new features.
    "pattern_analysis": string,              # Identify patterns in user-selected movies, including contrasts between selections.
    "serendipity_answer": string,            # Were the movies unexpected yet interesting?
    "likert_answer": int                     # Choose the option based on the 6-point Likert scale.
}}
"""


for fields in field_combinations:
    system_prompt = generate_prompt(fields)
    evaluation_name = "likert_" + "_".join(field.name.lower() for field in fields)

    print(f"Running evaluation for: {evaluation_name}")

    evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
    evaluator.evaluate_data(data, results_folder="./serendipity_results")
    
    print(f"Completed evaluation for: {evaluation_name}\n")
