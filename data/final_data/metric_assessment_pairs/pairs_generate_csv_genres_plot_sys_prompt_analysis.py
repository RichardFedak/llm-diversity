import csv
import json

input_file = 'pairs_final_movie_data.json'
output_file = 'movie_pairs_sys_prompt_genres_plot_analysis.csv'

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
        
        titles_A = {movie['title'] for movie in list_A}
        titles_B = {movie['title'] for movie in list_B}
        common_titles = titles_A & titles_B
        
        filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
        filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
        
        list_A_info = "\n".join([
            f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot: {movie['plot']}" 
            for movie in filtered_list_A
        ])
        list_B_info = "\n".join([
            f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot: {movie['plot']}" 
            for movie in filtered_list_B
        ])
        
        input_text = (
            f"List A:\n{list_A_info}\n\n"
            f"List B:\n{list_B_info}"
        )
        
        output_text = f"{selected_list}"
        
        writer.writerow({'input': input_text, 'output': output_text})

print(f"CSV file created: {output_file}")
