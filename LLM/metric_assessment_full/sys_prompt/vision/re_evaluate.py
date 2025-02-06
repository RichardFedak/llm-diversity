import os
import json
import time
import httpx
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Set evaluation name
EVALUATION_NAME = "covers_think"

# File paths
VALID_RESPONSES_FILE = f"valid_responses_{EVALUATION_NAME}.json"
RE_EVALUATED_RESPONSES_FILE = VALID_RESPONSES_FILE
FINAL_MOVIE_DATA_FILE = "final_movie_data.json"
CACHE_FILE = "image_cache.json"

# Load existing responses and movie data
with open(VALID_RESPONSES_FILE, "r") as f:
    response_data = json.load(f)

with open(FINAL_MOVIE_DATA_FILE, "r") as f:
    movie_data = json.load(f)

# Load cache for uploaded images
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

image_cache = load_cache()

# System prompt (same as the original)
sys_prompt = """
You are an assistant tasked with comparing three lists of movies. For each movie, the title, genres, plot, and cover are provided. 
Use your expertise and given information to determine key differences between the lists and identify the most diverse list.

Deliver your comparison and choice of the most diverse list in the following JSON format:
Note that:
    - The "comparison" field can be any custom object you create to compare and analyze key differences between the lists, such as visual themes, genres or create summaries of the lists containing the information to help determine the most diverse list.
    - The "most_diverse_list_reasoning" field should explain which list you perceive to be the most diverse and why.
    - The "most_diverse_list" field should contain the list you determine to be the most diverse, either 'A', 'B', or 'C'. 

{
    "comparison": object,                    
    "most_diverse_list_reasoning": string,    
    "most_diverse_list": string              
}
"""

# Configure the model
model = genai.GenerativeModel(
    system_instruction=sys_prompt,
    generation_config={"response_mime_type": "application/json"}
)

# Function to fetch and upload images
def fetch_save_and_upload_image(url, local_path):
    response = httpx.get(url)
    
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        uploaded_file = genai.upload_file(local_path)
        os.remove(local_path)
        
        return uploaded_file.name
    else:
        print(f"Failed to retrieve image from {url}")
        return None

# Function to process and upload images
def process_image_list(covers, list_name, cache):
    uploaded_images = []
    
    for idx, url in enumerate(covers):
        cache_key = f"{list_name}_{idx}"
        
        if cache_key in cache:
            cover_file = genai.get_file(cache[cache_key])
            uploaded_images.append(cover_file.name)
        else:
            local_path = f"{list_name}_{idx}.jpg"
            uploaded_uri = fetch_save_and_upload_image(url, local_path)
            if uploaded_uri:
                uploaded_images.append(uploaded_uri)
                cache[cache_key] = uploaded_uri  # Update cache

    return uploaded_images

# Process each response and retry where needed
re_evaluated_data = []
errors_encountered = 0
valid_fixes = 0
invalid_fixes = 0

for idx, item in enumerate(response_data):
    needs_fix = item["comparison"] == "X" or item["most_diverse_list_reasoning"] == "X" or item["output"] == "X"

    if not needs_fix:
        re_evaluated_data.append(item)  # Keep valid responses unchanged
        continue

    print(f"Retrying generation for index {idx}...")

    try:
        # Get original movie lists from final_movie_data.json
        movie_entry = movie_data[idx]  # Use the same index to retrieve data
        list_A = movie_entry['list_A']
        list_B = movie_entry['list_B']
        list_C = movie_entry['list_C']
        gold_most_diverse = movie_entry['selected_list']
        participation = movie_entry['participation']

        # Remove common titles across all lists
        titles_A = {movie['title'] for movie in list_A}
        titles_B = {movie['title'] for movie in list_B}
        titles_C = {movie['title'] for movie in list_C}
        common_titles = titles_A & titles_B & titles_C

        filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
        filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
        filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]

        # Extract movie covers
        list_A_covers = [movie['cover'] for movie in filtered_list_A]
        list_B_covers = [movie['cover'] for movie in filtered_list_B]
        list_C_covers = [movie['cover'] for movie in filtered_list_C]

        # Upload images and get URIs
        uploaded_list1 = process_image_list(list_A_covers, f"{idx}_list1", image_cache)
        uploaded_list2 = process_image_list(list_B_covers, f"{idx}_list2", image_cache)
        uploaded_list3 = process_image_list(list_C_covers, f"{idx}_list3", image_cache)

        # Construct the prompt
        list_A_info_str = "\n".join([f"{movie['title']} - Genres: ({movie['genres']}) - Plot: {movie['plot']}" for movie in filtered_list_A])
        list_B_info_str = "\n".join([f"{movie['title']} - Genres: ({movie['genres']}) - Plot: {movie['plot']}" for movie in filtered_list_B])
        list_C_info_str = "\n".join([f"{movie['title']} - Genres: ({movie['genres']}) - Plot: {movie['plot']}" for movie in filtered_list_C])

        prompt = (
            ["List A:\n"] + [list_A_info_str] + uploaded_list1 + 
            ["\n\nList B:\n"] + [list_B_info_str] + uploaded_list2 + 
            ["\n\nList C:\n"] + [list_C_info_str] + uploaded_list3
        )

        # Generate a new response
        response = model.generate_content(prompt)
        new_output = json.loads(response.text.strip())

        if all(key in new_output for key in ["comparison", "most_diverse_list_reasoning", "most_diverse_list"]):
            valid_fixes += 1
            item["comparison"] = new_output["comparison"]
            item["most_diverse_list_reasoning"] = new_output["most_diverse_list_reasoning"]
            item["output"] = new_output["most_diverse_list"]
            item["correct"] = new_output["most_diverse_list"] == gold_most_diverse
        else:
            invalid_fixes += 1

    except Exception as e:
        print(f"Error generating response for index {idx}: {e}")
        errors_encountered += 1

    re_evaluated_data.append(item)

    time.sleep(1)  # Prevent rate limiting

save_cache(image_cache)

# Save the new responses
with open(RE_EVALUATED_RESPONSES_FILE, "w") as f:
    json.dump(re_evaluated_data, f, indent=4)

print(f"\n--- Re-Evaluation Complete ---\nValid Fixes: {valid_fixes}, Errors: {errors_encountered}")
