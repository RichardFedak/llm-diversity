import csv
import json

input_file = 'pairs_final_movie_data.json'
output_file = 'movie_diversity_pairs_movie_plot.csv'

with open(input_file, 'r') as f:
    data = json.load(f)

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['input', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
        list_A = item['list_A']
        list_B = item['list_B']
        selected_list = item['selected_list']
        
        list_A_info = "\n".join([f"- {movie['title']} - Plot of the movie: {movie['plot']}" for movie in list_A])
        list_B_info = "\n".join([f"- {movie['title']} - Plot of the movie: {movie['plot']}" for movie in list_B])
        input_text = (
            "Consider the following two lists of movies 'A' and 'B'. Evaluate which list of movies is more diverse based on the information provided.\n\n"
            f"List A:\n{list_A_info}\n\n"
            f"List B:\n{list_B_info}\n\n"
            "After your evaluation, respond with only one letter:\n"
            "'A' if List A is more diverse.\n"
            "'B' if List B is more diverse.\n\n"
            "Do not provide any explanation, justification, or additional output. Strictly respond with the single letter: 'A' or 'B'."
        )
        
        output_text = f"{selected_list}"
        
        writer.writerow({'input': input_text, 'output': output_text})

print(f"CSV file created: {output_file}")
