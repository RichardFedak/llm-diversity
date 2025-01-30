import csv
import json
import ast

input_file = 'final_movie_data.json'

with open(input_file, 'r') as f:
    data = json.load(f)
count=0
for item in data:
    count += 1
print(count)