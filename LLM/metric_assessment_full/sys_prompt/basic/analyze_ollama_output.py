import os
import json

input_folder = "results_ollama"
output_folder = os.path.join(input_folder, "analysis")
os.makedirs(output_folder, exist_ok=True)

summary = []

for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not parse {filename}: {e}")
            continue

        evaluations = data.get("evaluations", [])
        correct = 0
        incorrect = 0
        incorrect_outputs = []
        listX_like_count = 0

        for item in evaluations:
            output = item.get("output", "")
            output_clean = output.strip() if isinstance(output, str) else ""

            if output_clean in {"A", "B", "C"}:
                correct += 1
            else:
                incorrect += 1
                incorrect_outputs.append(output)

            if (
                isinstance(output_clean, str)
                and len(output_clean) >= 6
                and output_clean[:4].upper() == "LIST"
                and output_clean[5].upper() in {"A", "B", "C"}
            ):
                listX_like_count += 1

        summary.append({
            "file": filename,
            "correct": correct,
            "incorrect": incorrect,
            "incorrect_outputs": incorrect_outputs,
            "LIST_X_outputs": listX_like_count
        })

output_path = os.path.join(output_folder, "analysis_summary.json")
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(summary, out_file, indent=4)

print(f"Analysis complete. Summary saved to {output_path}")
