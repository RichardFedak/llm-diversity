import os
import csv
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

img_folder = "img"
movies_csv = "movies.csv"
temp_csv = "movies_temp.csv"
links_csv = "links.csv"

movie_ids_in_img = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(".jpg")}

movie_to_imdb = {}
with open(links_csv, newline='', encoding="utf-8") as csvfile:
    for row in csv.DictReader(csvfile):
        mid = row["movieId"]
        if mid in movie_ids_in_img:
            movie_to_imdb[mid] = str(row["imdbId"]).zfill(7)

processed_ids = set()
if os.path.exists(temp_csv):
    with open(temp_csv, newline='', encoding="utf-8") as temp_file:
        temp_reader = csv.DictReader(temp_file)
        for row in temp_reader:
            processed_ids.add(row["movieId"])

with open(movies_csv, newline='', encoding="utf-8") as infile, \
     open(temp_csv, mode='a', newline='', encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames.copy()
    if "embedding" not in fieldnames:
        fieldnames.append("embedding")

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    if os.stat(temp_csv).st_size == 0:
        writer.writeheader()

    total = 87586
    curr = 0

    for row in reader:
        movie_id = row["movieId"]
        
        if movie_id in processed_ids:
            curr += 1
            continue

        curr += 1
        imdb_id = movie_to_imdb.get(movie_id)
        embedding = row.get("embedding", "").strip()

        if embedding:
            writer.writerow(row)
            continue

        if not imdb_id:
            row["embedding"] = "X"
            writer.writerow(row)
            continue

        try:
            row["embedding"] = model.encode([', '.join(row['genres']) + row['plot']])
            print(f"Embedding created: {row['title']}")
            print(f"{curr}/{total} - {(curr/total)*100:.2f}%")
        except Exception as e:
            print(f"Failed for {row['title']}: {e}")

        writer.writerow(row)
