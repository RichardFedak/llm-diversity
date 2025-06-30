import json

file_mappings = {
    "likert_elicitation_selected_per_user_genres_plot.json": "_system_prompt.txt",
}

for input_file, output_file in file_mappings.items():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        system_prompt = data.get("system_prompt", "")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(system_prompt)
