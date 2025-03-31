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
You are an assistant tasked with analyzing and predicting the user's perception of diversity.  
You are given data in JSON format, containing 3 Block with 6 Phases, each containin recommended movies and the selected ones the user preferred.
For each movie, you are given its: {field_text}.
You are also provided with a separate list of preferred movies that represent the user's taste, these movies were gathered during preference elicitation.

At the end of Block 0 and 1, you are provided with the user answer on the diversity question: Were the movies in this Block highly different from each other?
Answer is based on a 6-point Likert scale (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3). NOTE: Option 0 is not valid and can't be selected.  

Your task is to:
1. Analyze the preferred movies of the user
2. For each Block, access the diversity of the recommended movies in Block's phases.
3. For each Block, based on your analysis of diversity, interpret the user response they gave in Likert scale for the question: **Were the movies in this Block unexpected yet interesting?**
4. Finally, try to predict what answer (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3) they would give to the question for Block 2.

Deliver your evaluation in the following JSON format:  

{{
    "preferred_movies_analysis": string,     # Analyze the preferred movies. Describe what you can conclude from these movies.
    "block_0_analysis": string,              # Analyze the recommended movies.
    "block_0_diversity_reasoning",           # Interpret the answer the user gave based on the analysis.
    "block_1_analysis": string,              # Analyze the recommended movies.
    "block_1_diversity_reasoning",           # Interpret the answer the user gave based on the analysis. Compare it with previous block.
    "block_2_analysis": string,              # Analyze the recommended movies.
    "block_2_diversity_reasoning": string,   # Based on previous user answers try to interpret the reasoning, the user would give for question: **Were the movies in this block unexpected yet interesting?**
    "answer": int                            # Predict the answer based on the 6-point Likert scale. One number (strongly disagree = -3, disagree = -2, slightly disagree = -1, slightly agree = 1, agree = 2, strongly agree = 3). Consider previous answers by the user on Blocks 0 and 1.
}}
"""


for fields in field_combinations:
    with open("final_movie_data.json", 'r') as f:
        data = json.load(f) 

        system_prompt = generate_prompt(fields)
        evaluation_name = "likert_elicitation_selected_per_user_" + "_".join(field.name.lower() for field in fields)

        print(f"Running evaluation for: {evaluation_name}")

        evaluator = MovieEvaluator(api_key, evaluation_name, system_prompt, fields)
        evaluator.evaluate_data_per_user(data)
        
        print(f"Completed evaluation for: {evaluation_name}\n")
