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
    [MovieFields.GENRES],
    [MovieFields.PLOT],
    [MovieFields.GENRES, MovieFields.PLOT]
]

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
You are an assistant tasked with analyzing and predicting the user's perception of serendipity in 6 lists of movies.  
You are given data in JSON format, containing 3 blocks of 6 selection phases of movies the user preferred.
For each movie, you are given its: {field_text}.
You are also provided with a separate list of preferred movies that represent the user's taste.

For block 0 and 1, you are provided with the user response on the question on serendipity: Were the movies in this block unexpected yet interesting?
Response is based on a 6-point Likert scale (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3). NOTE: Option 0 is not valid, DON'T select it.  

Your task is to analyze the preferred movies of the user, analyze and assess serendipity for selected movies, and try to understand user responses on serendipity for blocks 0 and 1.
Then try to predict, what response the user would give for the same question in the block 2, based on Likert scale described.

Deliver your evaluation in the following JSON format:  

{{
    "preferred_movies_analysis": string,     # Analyze the preferred movies. Build a user profile based on these movies.
    "block_0_analysis": string,              # Analyze recommended and selected movies, if they were unexpected yet interesting for the user
    "block_0_serendipity_reasoning",         # Analyze the user answer
    "block_1_analysis": string,              # Analyze recommended and selected movies, if they were unexpected yet interesting for the user
    "block_1_serendipity_reasoning",         # Analyze the user answer
    "block_2_analysis": string,              # Analyze recommended and selected movies, if they were unexpected yet interesting for the user
    "block_2_serendipity_reasoning": string, # Based on previous user answers and selections try to predict the answer, the user would give for question: Were the movies in this block unexpected yet interesting?
    "answer": int                            # Predict the based on the 6-point Likert scale. One number (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3). Consider previous answers by the user on blocks 0 and 1.
}}
"""


for fields in field_combinations:
    with open("final_movie_data.json", 'r') as f:
        data = json.load(f) 

        system_prompt = generate_prompt(fields)
        evaluation_name = "likert_elicitation_selected_per_user_" + "_".join(field.name.lower() for field in fields)

        print(f"Running evaluation for: {evaluation_name}")

        evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
        evaluator.evaluate_data_per_user(data)
        
        print(f"Completed evaluation for: {evaluation_name}\n")
