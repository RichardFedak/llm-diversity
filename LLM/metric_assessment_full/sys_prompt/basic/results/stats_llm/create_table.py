import json
import os

def parse_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    name = data.get("name", "")
    acc = data.get("llm_output", {}).get("accuracy", {})
    selections = data.get("chosen_metrics", {}).get("percentages", {})
    
    return {
        "name": name,
        "llm-user accuracy": acc.get("output", ""),
        "llm-cf_ild accuracy": acc.get("cf_ild", ""),
        "llm-cb_ild accuracy": acc.get("cb_ild", ""),
        "llm-bin_div accuracy": acc.get("bin_div", ""),
        "cf_ild list selections": selections.get("CF-ILD", ""),
        "cb-ild list selections": selections.get("CB-ILD", ""),
        "bin_div list selections": selections.get("BIN-DIV", "")
    }

def format_float(v):
    if isinstance(v, float):
        val_str = f"{v:.4f}"
        if v > 0.45:
            return f"***{val_str}***"
        return val_str
    elif isinstance(v, int):
        val_str = f"{v}.0000"
        if v > 0.45:
            return f"***{val_str}***"
        return val_str
    else:
        # for empty string or other types, just return as string
        return str(v)

def main():
    files = [f for f in os.listdir(".") if f.endswith(".json")]
    rows = []
    
    for file in sorted(files):
        row = parse_json_file(file)
        rows.append(row)
    
    headers = [
        "name",
        "llm-user accuracy",
        "llm-cf_ild accuracy",
        "llm-cb_ild accuracy",
        "llm-bin_div accuracy",
        "cf_ild list selections",
        "cb-ild list selections",
        "bin_div list selections"
    ]
    
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["-"*len(h) for h in headers]) + "|")
    
    for r in rows:
        row_str = [format_float(r[h]) for h in headers]
        lines.append("| " + " | ".join(row_str) + " |")
    
    md_table = "\n".join(lines)
    
    with open("results.md", "w") as f:
        f.write(md_table)
    print("Markdown table saved to results.md")

if __name__ == "__main__":
    main()
