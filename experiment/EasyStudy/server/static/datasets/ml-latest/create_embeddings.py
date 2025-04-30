import os
import csv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

img_folder = "img"
movies_csv = "movies.csv"
links_csv = "links.csv"
embedding_file = "embeddings.npy"

movie_ids_in_img = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(".jpg")}

movie_to_imdb = {}
with open(links_csv, newline='', encoding="utf-8") as csvfile:
    for row in csv.DictReader(csvfile):
        mid = row["movieId"]
        if mid in movie_ids_in_img:
            movie_to_imdb[mid] = str(row["imdbId"]).zfill(7)

if os.path.exists(embedding_file):
    embedding_dict = np.load(embedding_file, allow_pickle=True).item()
else:
    embedding_dict = {}

with open(movies_csv, newline='', encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    total = 87586
    curr = 0

    for row in reader:
        movie_id = row["movieId"]
        curr += 1

        if movie_id in embedding_dict:
            continue

        if movie_id not in movie_to_imdb:
            continue

        try:
            input_text = "Genres: "+ ", ".join(row['genres']) + "\nPlot:" + row['plot']
            embedding = model.encode([input_text])[0]
            embedding_dict[movie_id] = embedding
            print(f"{curr}/{total} - Embedded: {row['title']}")
        except Exception as e:
            print(f"Failed for {row['title']}: {e}")

np.save(embedding_file, embedding_dict)
print(f"\nDone! Saved {len(embedding_dict)} embeddings to {embedding_file}")

embedding_dict = np.load(embedding_file, allow_pickle=True).item()

embedding_df = pd.DataFrame.from_dict(embedding_dict, orient='index')
embedding_df.index.name = 'movieId'
embedding_df.reset_index(inplace=True)
print(embedding_df.shape)
