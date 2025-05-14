import json

file_mappings = {
    "genres.json": "_system_prompt_basic.txt",
    "summary_genres.json": "_system_prompt_summary.txt",
    "standouts_genres.json": "_system_prompt_standouts.txt"
}

for input_file, output_file in file_mappings.items():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        system_prompt = data.get("system_prompt", "")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(system_prompt)
