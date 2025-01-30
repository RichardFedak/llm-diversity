import csv
import json
import ast

input_file = 'final_movie_data.json'
output_file = 'tuned_full.csv'

with open(input_file, 'r') as f:
    data = json.load(f)

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['input', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
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
        
        list_A_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_A])
        list_B_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_B])
        list_C_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_C])

        input_text = (
            """You are an evaluator. You are given three lists of movies. Your task is to order the lists from least to most diverse based on the information given.

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
            """For example, if the least diverse list is B, then C, and the most diverse is A, the output should be: B,C,A

Do NOT provide explanations, comments, or any additional text — only the final output in the required format representing the order from least to most diverse list.
"""
            )
        res = [int(x.strip()) for x in ordering[1:-1].split(',')]
        output_text = output_text = ",".join(chr(65 + x) for x in res)
        
        writer.writerow({'input': input_text, 'output': output_text})

print(f"CSV file created: {output_file}")
