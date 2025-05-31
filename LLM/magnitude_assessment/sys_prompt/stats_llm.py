import json
import os
from collections import defaultdict
from scipy.stats import spearmanr, wasserstein_distance
import numpy as np

RESULTS_FOLDER = "results_ollama/"
STATS_FOLDER = os.path.join(RESULTS_FOLDER, "stats_llm/")
DATASET_FILE = "final_movie_data.json"

os.makedirs(STATS_FOLDER, exist_ok=True)

def load_dataset():
    """Loads the dataset file into a dictionary."""
    try:
        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in dataset file: {e}")
        return {}

def analyze_file(file_path, dataset):
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
    emd_distances = []
    spearman_correlations = []
        
    selected_list_to_index = {"A": 0, "B": 1, "C": 2}
    list_metrics_value_counts = defaultdict(int)
    
    for idx, response in enumerate(evaluations):
        participation = response["participation"]
        llm_final_ordering = response["final_ordering"]
        llm_approx_alphas = response["approx_scores"]
        dataset_entry = dataset[idx]

        if not dataset_entry:
            print(f"Warning: No corresponding dataset entry found")
            total_evaluations -= 1
            continue

        used_metric = dataset_entry["compare_alphas_metric"]
        user_approx_alphas = [float(x.strip()) for x in dataset_entry["approx_alphas"][1:-1].split(',')]
        user_final_ordering = [int(x.strip()) for x in dataset_entry["selected"][1:-1].split(',')]
        cf_ild = dataset_entry["cf_ild"]
        cb_ild = dataset_entry["cb_ild"]
        bin_div = dataset_entry["bin_div"]

        if llm_final_ordering == user_final_ordering:
            correct_evals["output"] += 1
            list_metrics_value_counts[used_metric] += 1
        if llm_final_ordering == cf_ild:
            correct_evals["cf_ild"] += 1
        if llm_final_ordering == cb_ild:
            correct_evals["cb_ild"] += 1
        if llm_final_ordering == bin_div:
            correct_evals["bin_div"] += 1
        
        emd_distance = wasserstein_distance(llm_approx_alphas, user_approx_alphas)
        emd_distances.append(emd_distance)

        res = spearmanr(llm_approx_alphas, user_approx_alphas)
        spearman_correlations.append(res.statistic)

    for key in correct_evals:
        accuracy[key] = (
            round(correct_evals[key] / total_evaluations if total_evaluations > 0 else 0.0, 4)
        )
    
    mean_emd = np.mean(emd_distances)
    mean_spearman = np.nanmean(spearman_correlations)
        
    summary_data = {
        "name": eval_name,
        "total_evaluations": total_evaluations,
        "llm_output": {
            "correct_evaluations": correct_evals,
            "accuracy": accuracy,
        },
        "correct_output_metric_counts": dict(list_metrics_value_counts),
        "emd": mean_emd,
        "spearman": mean_spearman,
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
    
    dataset = load_dataset()
    if not dataset:
        print("Skipping file processing due to missing dataset.")
        return
    
    files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]
    if not files:
        print("No JSON files found in the results folder.")
        return
    
    for file in files:
        file_path = os.path.join(RESULTS_FOLDER, file)
        analyze_file(file_path, dataset)

if __name__ == "__main__":
    process_all_files()
