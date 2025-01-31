import time
import os
import csv
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

sys_prompt = """
You are an AI assistant specializing in analyzing and summarizing movie lists.
Your task is to generate a JSON summary for a given movie list, capturing both a natural language description and structured diversity metrics.

How to create a summary:

1. Popularity Diversity
Assign a score based on whether the list contains mainly niche/independent films or blockbuster movies or a mix of both.  
    - A score of 1 represents most of the movies are niche/independent or only blockbusters.
    - A score of 2 represents few movies from one side of the popularity spectrum and more from the other.
    - A score of 3 represents a balance of both blockbuster movies and niche/independent films.
    - For every movie that comes from a different side of the popularity spectrum (blockbuster or indie), increment the score.

2. Genre Diversity
Assess the variety of genres represented in the list. 
    - A score of 1 represents Low Genre Diversity, where most of the different movies in the list share similar genres or genres are repeated across most movies.
    - A score of 2 represents Moderate Genre Diversity, where the movies in the list cover a mix of genres but some genres are repeated.
    - A score of 3 represents High Genre Diversity, where most of the different movies in the list span multiple different genres or most of the movies belong to unique genres.
    - The more varied the genres across the different movies in the list, the higher the score.

Example of High Genre Diversity pair:  
    - The Matrix (Action, Sci-Fi, Thriller)  
    - Toy Story (Animation, Adventure, Comedy)

Example of Low Genre Diversity pair:
    - The Dark Knight (Action, Drama, Thriller)  
    - Inception (Action, Crime, Thriller, Drama)

3. Theme Diversity 
Evaluate the thematic range of the movies in the list. 
    - A score of 1 represents Low Theme Diversity, where most of the different movies in the list revolve around a limited set of themes, such as only love or only war.
    - A score of 2 represents Moderate Theme Diversity, where the different movies cover a mix of themes but some themes are repeated across multiple movies.
    - A score of 3 represents High Theme Diversity, where the most of the different movies cover a mix of diverse themes like love, peace, revenge, isolation, and more.
    - The more varied and expansive the themes across the movies in the list, the higher the score.

Example of High Theme Diversity pair:  
- John Wick (Revenge)
- La La Land (Love)

Example of Low Theme Diversity pair:  
- Cast Away (Isolation)
- The Martian (Isolation)

4. Time Span: Determine the earliest and latest movie release years.
5. Franchise Inclusion: Check if the list includes at least two movies from the same franchise.

Deliver your summary in the following JSON format:

{
    "popularity_diversity": {
        "value": "number",  # An integer from 1 (low) to 3 (high)
        "reasoning": "string"
    },
    "genre_diversity": {
        "value": "number",  # An integer from 1 (low) to 3 (high)
        "reasoning": "string"
    },
    "theme_diversity": {
        "value": "number",  # An integer from 1 (low) to 3 (high)
        "reasoning": "string"
    },
    "time_span": {
        "value": "string",  # e.g. "1990 - 2020"
        "reasoning": "string"
    },
    "franchise_inclusion": {
        "value": "boolean",  # True if the list includes at least two movies from the same franchise, False otherwise
        "reasoning": "string"
    }
}

"""

model = genai.GenerativeModel(
    system_instruction=sys_prompt, 
    generation_config={"response_mime_type": "application/json"},
    )

file_name = "final_movie_data.json"

REQUEST_INTERVAL = 4

with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

def transform_list(original_list, prompt_text):
    time.sleep(REQUEST_INTERVAL)
    try:
        response = model.generate_content(prompt_text)
        output = json.loads(response.text.strip())
        if all(key in output for key in ["genre_diversity", "theme_diversity", "time_span", "franchise_inclusion", "popularity_diversity"]):
            return {
                "summary": output,
                "items": original_list
            }
        return {
            "summary": {},
            "items": original_list
        }
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            "summary": {},
            "items": original_list
        }
idx=0
for item in data:
    print(idx/len(data))
    idx+=1

    list_A = item['list_A']
    list_B = item['list_B']
    list_C = item['list_C']

    list_A_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in list_A])
    list_B_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in list_B])
    list_C_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in list_C])

    item["list_A"] = transform_list(item["list_A"], list_A_info)
    item["list_B"] = transform_list(item["list_B"], list_B_info)
    item["list_C"] = transform_list(item["list_C"], list_C_info)


updated_file_name = "final_movie_data_with_summary.json"
with open(updated_file_name, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated JSON saved as {updated_file_name}")
