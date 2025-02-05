import csv
import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

input_file = 'final_movie_data.json'
output_file = 'tuned_reasoning.csv'

sys_prompt = """
You are an evaluator given three lists of movies with titles provided.
You will also receive a correct identification of the most diverse list of movies. 
Your task is to write a reasoning based on that identification, describing why the identified list is the most diverse.
The identification is always correct as it reflects the perception of real users so always try to find a reasoning that fits the identification.

Use the following JSON schema for your response:
{
    "reasoning": str,         # Explain your evaluation of the lists and why the identified list is the most diverse.
    "most_diverse_list": str  # The list identifier ('A', 'B', or 'C') that is the most diverse.
}

"""

reasoning_model = genai.GenerativeModel(system_instruction=sys_prompt, generation_config={"response_mime_type": "application/json"})

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open(input_file, 'r') as f:
    data = json.load(f)

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['input', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
        current_time = time.time()
        if requests_made >= MAX_REQUESTS_PER_MINUTE:
            elapsed_time = current_time - last_request_time
            if elapsed_time < 60:
                time_to_wait = 80 - elapsed_time
                print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                time.sleep(time_to_wait)
            requests_made = 0
            last_request_time = time.time()
        list_A = item['list_A']
        list_B = item['list_B']
        list_C = item['list_C']
        gold_most_diverse = item['selected_list']
        participation = item['participation']

        titles_A = {movie['title'] for movie in list_A}
        titles_B = {movie['title'] for movie in list_B}
        titles_C = {movie['title'] for movie in list_C}
        common_titles = titles_A & titles_B & titles_C
        
        filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
        filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
        filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]
        
        list_A_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_A])
        list_B_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_B])
        list_C_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_C])

        prompt = (
            f"List A:\n{list_A_info}\n\n"
            f"List B:\n{list_B_info}\n\n"
            f"List C:\n{list_C_info}\n\n"
            f"The most diverse list: {gold_most_diverse}"
        )

        reasoning_response = reasoning_model.generate_content(prompt)
        requests_made += 1
        print(requests_made)
        time.sleep(REQUEST_INTERVAL)
        reasoning_output = reasoning_response.text.strip()

        input_text = (
"""You are given three lists of movies with titles provided. Use your expertise to assess the lists and choose the one with the most diverse collection of movies.
Use the following JSON schema for your response:
{
    "reasoning": str,         # Explain your evaluation of the lists and why the given ordering is correct.
    "final_ordering": array   # Array, ordering of the lists 'A', 'B' and 'C' starting from the least diverse list.
}

"""
f"List A:\n{list_A_info}\n\n"
f"List B:\n{list_B_info}\n\n"
f"List C:\n{list_C_info}\n\n"
            )
        
        writer.writerow({'input': input_text, 'output': reasoning_output})
        

print(f"CSV file created: {output_file}")
