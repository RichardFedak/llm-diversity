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

For each list given, describe the serendipity by telling if there are any selected movies that stand out compared to the movies user already selected before (preferred, selected in previous lists). Name the movies that stand out or write that selected movies were not unexpected.

Summarize the serendipity across the lists by answering the question: **Were the movies across the lists unexpected yet interesting?**  
Consider variations in genre, themes, time periods, and other relevant aspects. Also, analyze whether the movies align with or diverge from the user's preferred and already selected movies.  

Then, based on a 6-point Likert scale (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3), choose the option for the whole batch of lists. Option 0 is not valid, DON'T select it.  

Deliver your descriptions of lists, comparison with preferred movies, overall serendipity summarization, and serendipity score for the entire batch in the following JSON format:  

{{
    "preferred_movies_analysis": string,     # Analyze the preferred movies and describe their common themes, genres, and characteristics. Build a user profile based on these movies.
    "list_A_selections_description": string,
    "list_B_selections_description": string,
    "list_C_selections_description": string,
    "list_D_selections_description": string,
    "list_E_selections_description": string,
    "list_F_selections_description": string,
    "serendipity_summarization": string,     # Answer the question: Were the movies across the lists unexpected yet interesting? Consider if there were more lists with unexpected and interesting movies (agree) or more lists that contained movies that were not interesting (disagree).
    "answer": int                            # Choose the option based on the 6-point Likert scale.
}}
"""


for fields in field_combinations:
    system_prompt = generate_prompt(fields)
    evaluation_name = "likert_elicitation_selected_" + "_".join(field.name.lower() for field in fields)

    print(f"Running evaluation for: {evaluation_name}")

    evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields, include_summary=True)
    evaluator.evaluate_data(data)
    
    print(f"Completed evaluation for: {evaluation_name}\n")
