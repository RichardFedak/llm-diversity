import json
from collections import defaultdict

with open("../metric_assessment_full/final_movie_data.json", "r") as f:
    metric_data = json.load(f)

with open("../magnitude_assessment/final_movie_data.json", "r") as f:
    magnitude_data = json.load(f)

with open("../recommendations/final_movie_data.json", "r") as f:
    recommendation_data = json.load(f)

# Magnitude data conatains 2 objects for each participant...
magnitude_joined = defaultdict(list)
for evaluation in magnitude_data:
    magnitude_joined[evaluation["participation"]].append(evaluation)

joined_full = []

for evaluation in metric_data:
    participation = evaluation["participation"]
    joined_data = {
        "participation": participation,
        "metric_data": evaluation
    }

    joined_data["magnitude_data"] = magnitude_joined.get(participation)
    joined_data["recommendation_data"] = next((e for e in recommendation_data if e["participation"] == participation))

    joined_full.append(joined_data)

with open("data_joined.json", "w") as f:
    json.dump(joined_full, f, indent=4)