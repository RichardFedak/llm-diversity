import os
import json

def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        name = data.get("name", "")
        accuracy = data.get("llm_output", {}).get("accuracy", {}).get("output", "")
        pearson = data.get("pearson", "")
        binary = data.get("binary_accuracy", "")

        # Ensure rounding to 2 decimals if values are numeric
        if isinstance(accuracy, (int, float)):
            accuracy = f"{round(accuracy, 2):.2f}"
        else:
            accuracy = ""

        if isinstance(pearson, (int, float)):
            pearson = f"{round(pearson, 2):.2f}"
        else:
            pearson = ""
        
        if isinstance(binary, (int, float)):
            binary = f"{round(binary, 2):.2f}"
        else:
            binary = ""

        return name, accuracy, pearson, binary

def generate_markdown_table(entries):
    lines = [
        "| Name | LLM-User Accuracy | Pearson | Binary |",
        "|------|-------------------|---------|--------|"
    ]
    for name, acc, pearson, binary in entries:
        lines.append(f"| {name} | {acc} | {pearson} | {binary}")
    return "\n".join(lines)

entries = []
for filename in os.listdir("."):
    if filename.endswith(".json"):
        try:
            entry = parse_json_file(filename)
            entries.append(entry)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

markdown_content = generate_markdown_table(entries)

with open("summary.md", "w") as f:
    f.write(markdown_content)
