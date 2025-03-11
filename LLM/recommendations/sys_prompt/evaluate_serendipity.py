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
You are an assistant tasked with predicting the user's perception of serendipity in 6 lists of movies.  
For each movie, you are given its: {field_text}.
Additionally, you are provided with a separate list of preferred movies that represent the user's taste.  

Your task is to analyze the movies in each of the 6 lists and compare them to each other as well as to the preferred movies.  
Based on this analysis, predict how the user would perceive the serendipity of these lists.  

Summarize the serendipity across the lists by answering the question: **Were the movies across the lists unexpected yet interesting?**  
Consider variations in genre, themes, time periods, and other relevant aspects. Also, analyze whether the movies align with or diverge from the user's preferred movies.  

Then, based on a 6-point Likert scale (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3), choose the option for the whole batch of lists. Option 0 is not valid, DON'T select it.  

Deliver your descriptions of lists, comparison with preferred movies, overall serendipity summarization, and serendipity score for the entire batch in the following JSON format:  

{{
    "list_A_description": string,            # Describe the serendipity of the movies in list A, noting differences and similarities with the preferred movies.
    "list_B_description": string,            # Describe the serendipity of the movies in list B, noting differences and similarities with the preferred movies.
    "list_C_description": string,            # Describe the serendipity of the movies in list C, noting differences and similarities with the preferred movies.
    "list_D_description": string,            # Describe the serendipity of the movies in list D, noting differences and similarities with the preferred movies.
    "list_E_description": string,            # Describe the serendipity of the movies in list E, noting differences and similarities with the preferred movies.
    "list_F_description": string,            # Describe the serendipity of the movies in list F, noting differences and similarities with the preferred movies.
    "preferred_movies_analysis": string,     # Analyze the preferred movies and describe their common themes, genres, and characteristics.
    "serendipity_summarization": string,     # Answer the question: Were the movies across the lists unexpected yet interesting? Consider serendipity within the lists and in relation to the preferred movies.
    "answer": int                            # Choose the option based on the 6-point Likert scale.
}}
"""


for fields in field_combinations:
    system_prompt = generate_prompt(fields)
    evaluation_name = "likert_elicitation_" + "_".join(field.name.lower() for field in fields)

    print(f"Running evaluation for: {evaluation_name}")

    evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
    evaluator.evaluate_data(data)
    
    print(f"Completed evaluation for: {evaluation_name}\n")
