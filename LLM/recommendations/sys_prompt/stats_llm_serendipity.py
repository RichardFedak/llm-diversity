import json
import os
from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np

RESULTS_FOLDER = "serendipity_results/"
STATS_FOLDER = os.path.join(RESULTS_FOLDER, "stats_llm/")

os.makedirs(STATS_FOLDER, exist_ok=True)

def analyze_file(file_path):
    """Processes a single JSON results file and generates a summary."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return
    
    eval_name = data["name"]
    evaluations = data["evaluations"]
    total_evaluations = len(evaluations)
    
    correct_evals = {"output": 0}
    accuracy = {"output": 0.0}
    serendipity_score_counts = {
        "user": {},
        "llm": {}
    }
    gold_outputs = []
    llm_outputs =[]
            
    for idx, response in enumerate(evaluations):

        gold_output = response["gold"]
        llm_output = response["serendipity_score"]

        if gold_output == llm_output:
            correct_evals["output"] += 1
        
        serendipity_score_counts["user"][gold_output] = serendipity_score_counts["user"].get(gold_output, 0) + 1
        serendipity_score_counts["llm"][llm_output] = serendipity_score_counts["llm"].get(llm_output, 0) + 1
        
        gold_outputs.append(gold_output)
        llm_outputs.append(llm_output)

    for key in correct_evals:
        accuracy[key] = (
            round(correct_evals[key] / total_evaluations if total_evaluations > 0 else 0.0, 4)
        )
    
    pearson_corr = pearsonr(gold_outputs, llm_outputs).statistic
        
    summary_data = {
        "name": eval_name,
        "total_evaluations": total_evaluations,
        "serendipity_score_counts": serendipity_score_counts,
        "llm_output": {
            "correct_evaluations": correct_evals,
            "accuracy": accuracy,
        },
        "pearson": pearson_corr,
    }
    
    output_summary_file = os.path.join(
        STATS_FOLDER, f"{os.path.basename(file_path)}"
    )
    try:
        with open(output_summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary saved: {output_summary_file}")
    except IOError as e:
        print(f"Error: Could not write to file {output_summary_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving {output_summary_file}: {e}")

def process_all_files():
    """Processes all JSON files in the results folder."""
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Error: Folder '{RESULTS_FOLDER}' not found.")
        return
    
    files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]
    if not files:
        print("No JSON files found in the results folder.")
        return
    
    for file in files:
        file_path = os.path.join(RESULTS_FOLDER, file)
        analyze_file(file_path)

if __name__ == "__main__":
    process_all_files()
