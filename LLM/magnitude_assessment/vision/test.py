import os
import base64
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

list1_covers = [
    "https://m.media-amazon.com/images/M/MV5BMDliOTIzNmUtOTllOC00NDU3LWFiNjYtMGM0NDc1YTMxNjYxXkEyXkFqcGdeQXVyNTM3NzExMDQ@.jpg",
    "https://m.media-amazon.com/images/M/MV5BYjVhMmM3ZTMtNzIzOS00YTY4LTkxNTAtOTA5Mjk3YzMwMzA2XkEyXkFqcGdeQXVyNDE0OTU3NDY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTY5MzYzNjc5NV5BMl5BanBnXkFtZTYwNTUyNTc2.jpg",
    "https://m.media-amazon.com/images/M/MV5BMGNlMGZiMmUtZjU0NC00MWU4LWI0YTgtYzdlNGVhZGU4NWZlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BOTA5NDZlZGUtMjAxOS00YTRkLTkwYmMtYWQ0NWEwZDZiNjEzXkEyXkFqcGdeQXVyMTMxODk2OTU@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTYyOTk4MjIxOF5BMl5BanBnXkFtZTcwMzk1NTUyMQ@@.jpg",
    "https://m.media-amazon.com/images/M/MV5BNjUxYzliZjUtZTcyZS00NjQ2LTkxZWMtMzg5OWUxZGZkODFjXkEyXkFqcGdeQXVyMTUzMDUzNTI3.jpg",
    "https://m.media-amazon.com/images/M/MV5BOTA5MzQ3MzI1NV5BMl5BanBnXkFtZTgwNTcxNTYxMTE@.jpg"
]

list2_covers = [
    "https://m.media-amazon.com/images/M/MV5BM2JkNGU0ZGMtZjVjNS00NjgyLWEyOWYtZmRmZGQyN2IxZjA2XkEyXkFqcGdeQXVyNTIzOTk5ODM@.jpg",
    "https://m.media-amazon.com/images/M/MV5BYTIyMDFmMmItMWQzYy00MjBiLTg2M2UtM2JiNDRhOWE4NjBhXkEyXkFqcGdeQXVyNjU0OTQ0OTY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BNjM0NTc0NzItM2FlYS00YzEwLWE0YmUtNTA2ZWIzODc2OTgxXkEyXkFqcGdeQXVyNTgwNzIyNzg@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTc5OTk4MTM3M15BMl5BanBnXkFtZTgwODcxNjg3MDE@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTczNTI2ODUwOF5BMl5BanBnXkFtZTcwMTU0NTIzMw@@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTY5MzYzNjc5NV5BMl5BanBnXkFtZTYwNTUyNTc2.jpg",
    "https://m.media-amazon.com/images/M/MV5BYjVhMmM3ZTMtNzIzOS00YTY4LTkxNTAtOTA5Mjk3YzMwMzA2XkEyXkFqcGdeQXVyNDE0OTU3NDY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BOTY4YjI2N2MtYmFlMC00ZjcyLTg3YjEtMDQyM2ZjYzQ5YWFkXkEyXkFqcGdeQXVyMTQxNzMzNDI@.jpg"
]

list3_covers = [
    "https://m.media-amazon.com/images/M/MV5BNjM0NTc0NzItM2FlYS00YzEwLWE0YmUtNTA2ZWIzODc2OTgxXkEyXkFqcGdeQXVyNTgwNzIyNzg@.jpg",
    "https://m.media-amazon.com/images/M/MV5BM2JkNGU0ZGMtZjVjNS00NjgyLWEyOWYtZmRmZGQyN2IxZjA2XkEyXkFqcGdeQXVyNTIzOTk5ODM@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTY5MzYzNjc5NV5BMl5BanBnXkFtZTYwNTUyNTc2.jpg",
    "https://m.media-amazon.com/images/M/MV5BYTIyMDFmMmItMWQzYy00MjBiLTg2M2UtM2JiNDRhOWE4NjBhXkEyXkFqcGdeQXVyNjU0OTQ0OTY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTc5OTk4MTM3M15BMl5BanBnXkFtZTgwODcxNjg3MDE@.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTczNTI2ODUwOF5BMl5BanBnXkFtZTcwMTU0NTIzMw@@.jpg",
    "https://m.media-amazon.com/images/M/MV5BYjVhMmM3ZTMtNzIzOS00YTY4LTkxNTAtOTA5Mjk3YzMwMzA2XkEyXkFqcGdeQXVyNDE0OTU3NDY@.jpg",
    "https://m.media-amazon.com/images/M/MV5BOTY4YjI2N2MtYmFlMC00ZjcyLTg3YjEtMDQyM2ZjYzQ5YWFkXkEyXkFqcGdeQXVyMTQxNzMzNDI@.jpg"
]

def fetch_save_and_upload_image(url, local_path):
    # Send a request to get the image
    response = httpx.get(url)
    
    if response.status_code == 200:
        # Save the image to a local file
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        # Upload the image via the File API
        uploaded_file = genai.upload_file(local_path)
        
        # Clean up by deleting the local file after uploading
        os.remove(local_path)
        
        return uploaded_file
    else:
        print(f"Failed to retrieve image from {url}")
        return None

def process_image_list(covers, list_name):
    uploaded_images = []
    for idx, url in enumerate(covers):
        local_path = f"{list_name}_{idx}.jpg"  # Local file name for the image
        uploaded_uri = fetch_save_and_upload_image(url, local_path)
        if uploaded_uri:
            uploaded_images.append(uploaded_uri)
    return uploaded_images

uploaded_list1 = process_image_list(list1_covers, "list1")
uploaded_list2 = process_image_list(list2_covers, "list2")
uploaded_list3 = process_image_list(list3_covers, "list3")

prompt = """

Analyze these three lists of movie covers based on diversity. 
Each list has 3 movie covers.
Tell me which list is the most diverse and explain why.
Movie diversity refers to the variety of genres, themes, visual styles, and representations present within a collection of films.
"""

response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(
     ["List 1:\n"] + uploaded_list1 + ["\n\nList 2:\n"] + uploaded_list2 + ["\n\nList 3:\n"] + uploaded_list3 + [prompt]
)

# Print the result
print(response.text)