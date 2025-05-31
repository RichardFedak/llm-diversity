import json
import os
from collections import defaultdict

RESULTS_FOLDER = "results_ollama/"
STATS_FOLDER = os.path.join(RESULTS_FOLDER, "stats_llm_sanitized/")
DATASET_FILE = "final_movie_data.json"

os.makedirs(STATS_FOLDER, exist_ok=True)

def load_dataset():
    """Loads the dataset file into a dictionary."""
    try:
        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)
        return {item["participation"]: item for item in dataset}
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in dataset file: {e}")
        return {}

def is_output_list_X_correct(output, gold):
    """Checks if the output is in the format of 'LIST X ...'."""
    if isinstance(output, str) and len(output) >= 6:
        return output[:4].upper() == "LIST" and output[5].upper() in {"A", "B", "C"} and output[5].upper() == gold
    return False

def analyze_file(file_path, dataset_dict):
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
    
    correct_evals = {"output": 0, "cf_ild": 0, "cb_ild": 0, "bin_div": 0}
    accuracy = {"output": 0.0, "cf_ild": 0.0, "cb_ild": 0.0, "bin_div": 0.0}
    
    selected_list_to_index = {"A": 0, "B": 1, "C": 2}
    list_metrics_value_counts = defaultdict(int)
    
    for response in evaluations:
        participation = response["participation"]
        gold = response["gold"]
        output = response["output"]
        dataset_entry = dataset_dict.get(participation)

        if not dataset_entry:
            print(f"Warning: No corresponding dataset entry found for participation {participation}. Skipping.")
            total_evaluations -= 1
            continue

        cf_ild = dataset_entry["cf_ild"]
        cb_ild = dataset_entry["cb_ild"]
        bin_div = dataset_entry["bin_div"]
        list_metrics = dataset_entry["list_metrics"]

        if output and output == gold or is_output_list_X_correct(output, gold):
            correct_evals["output"] += 1
        if cf_ild and output == cf_ild:
            correct_evals["cf_ild"] += 1
        if cb_ild and output == cb_ild:
            correct_evals["cb_ild"] += 1
        if bin_div and output == bin_div:
            correct_evals["bin_div"] += 1

        if output in selected_list_to_index and list_metrics:
            index = selected_list_to_index[output]
            if 0 <= index < len(list_metrics):
                metric = list_metrics[index]
                list_metrics_value_counts[metric] += 1
    
    for key in correct_evals:
        accuracy[key] = (
            round(correct_evals[key] / total_evaluations if total_evaluations > 0 else 0.0, 4)
        )
    
    chosen_metrics_percentages = {
        metric: round(count / total_evaluations, 4)
        for metric, count in list_metrics_value_counts.items()
    }
    
    summary_data = {
        "name": eval_name,
        "total_evaluations": total_evaluations,
        "llm_output": {
            "correct_evaluations": correct_evals,
            "accuracy": accuracy,
        },
        "chosen_metrics": {
            "counts": dict(list_metrics_value_counts),
            "percentages": chosen_metrics_percentages,
        },
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
    
    dataset_dict = load_dataset()
    if not dataset_dict:
        print("Skipping file processing due to missing dataset.")
        return
    
    files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]
    if not files:
        print("No JSON files found in the results folder.")
        return
    
    for file in files:
        file_path = os.path.join(RESULTS_FOLDER, file)
        analyze_file(file_path, dataset_dict)

if __name__ == "__main__":
    process_all_files()
