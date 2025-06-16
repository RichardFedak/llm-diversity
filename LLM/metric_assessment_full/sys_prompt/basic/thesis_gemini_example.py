import os, json
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

system_prompt = """
You are a helpful assistant that provides information about movies.
Allways provide the answer in the following JSON format:
{{
    "movie_description": string,   # A short description of the movie
    "release_year": int,           # The year the movie was released
}}
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-001",
    system_instruction=system_prompt,
    generation_config={"response_mime_type": "application/json"}
)

user_prompt = "Please, tell me something about the movie 'Avatar'."
raw_response = model.generate_content(user_prompt)

data = json.loads(raw_response.text.strip())
print(data["movie_description"]) # In the year 2154, a paraplegic marine dispatched to the moon Pandora ...
print(data["release_year"]) # 2009
