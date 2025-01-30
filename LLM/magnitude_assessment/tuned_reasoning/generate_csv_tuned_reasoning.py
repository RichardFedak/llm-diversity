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
You are an evaluator given three lists of movies with titles and genres provided.
You will also receive a correct ordering of the movies based on diversity, from least diverse (first) to most diverse (last). Your task is to write a reasoning based on that ordering, describing why the ordering is correct.
The ordering is always correct as it reflects the perception of real users so allways try to find a reasoning that fits the ordering.

Use the following JSON schema for your response:
{
    "reasoning": str,         # Explain your evaluation of the lists and why the given ordering is correct.
    "final_ordering": array   # Array, ordering of the lists 'A', 'B' and 'C' starting from the least diverse list.
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
        list_A = item['list1']
        list_B = item['list2']
        list_C = item['list3']
        ordering = item['selected']

        titles_A = {movie['title'] for movie in list_A}
        titles_B = {movie['title'] for movie in list_B}
        titles_C = {movie['title'] for movie in list_C}
        common_titles = titles_A & titles_B & titles_C
        
        filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
        filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
        filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]
        
        list_A_info = "\n".join([f"- Movie: {movie['title']} - Genres of the movie: {movie['genres']}" for movie in filtered_list_A])
        list_B_info = "\n".join([f"- Movie: {movie['title']} - Genres of the movie: {movie['genres']}" for movie in filtered_list_B])
        list_C_info = "\n".join([f"- Movie: {movie['title']} - Genres of the movie: {movie['genres']}" for movie in filtered_list_C])

        res = [int(x.strip()) for x in ordering[1:-1].split(',')]
        output_text = ",".join(chr(65 + x) for x in res)
        prompt = (
            f"List A:\n{list_A_info}\n\n"
            f"List B:\n{list_B_info}\n\n"
            f"List C:\n{list_C_info}\n\n"
            f"Correct ordering: {output_text}"
        )

        reasoning_response = reasoning_model.generate_content(prompt)
        requests_made += 1
        print(requests_made)
        time.sleep(REQUEST_INTERVAL)
        reasoning_output = reasoning_response.text.strip()

        input_text = (
            """You are an evaluator. You are given three lists of movies with titles and genres provided. Your task is to order the lists from least to most diverse based on the information given and your knowledge.

Focus on the following two aspects:
1. Genres - This is the most important factor in determining diversity.
2. Plot and Themes - Only if the genres are almost the same or overlap significantly between the lists, use plot and themes to differentiate them.

1. Focus on Genres:
    - Assess the variety and uniqueness of genres in each list.
    - Identify how diverse the genres are within each list and across lists.

2. Focus on Plot and Themes:
    - If genres are similar or overlap significantly between the lists:
    - Analyze the plots to find nuanced differences in storytelling approaches.
    - Evaluate the overarching themes and how they contribute to the distinctiveness of each list.

"""
            f"List A:\n{list_A_info}\n\n"
            f"List B:\n{list_B_info}\n\n"
            f"List C:\n{list_C_info}\n\n"
"""Use the following JSON schema for your response:
{
    "reasoning": str,         # Explain your evaluation of the lists and why the given ordering is correct.
    "final_ordering": array   # Array, ordering of the lists 'A', 'B' and 'C' starting from the least diverse list.
}
"""
            )
        
        writer.writerow({'input': input_text, 'output': reasoning_output})

print(f"CSV file created: {output_file}")
