import json
from collections import defaultdict

EVAL_NAME = "covers_think"

VALID_RESPONSES_FILE = "valid_responses_" + EVAL_NAME + ".json"
DATASET_FILE = "final_movie_data.json"
OUTPUT_SUMMARY_FILE = "summary_data_" + EVAL_NAME + ".json"

def transform_to_indices(value_list):
    """
    Transforms a list of values into a ranking-based list of indices.
    Equivalent values are replaced with 'X'.
    """
    sorted_values = sorted(set(value_list), reverse=True)
    index_map = {v: i for i, v in enumerate(sorted_values) if sorted_values.count(v) == 1}
    return [index_map.get(v, 'X') for v in value_list]

def compare_with_x_ignore(response_output, transformed_metric):
    """
    Compares response output with transformed metric, ignoring 'X' values.
    """
    return all(ro == tm or tm == 'X' for ro, tm in zip(response_output, transformed_metric))

def analyze_data(valid_responses_file, dataset_file, output_summary_file):
    try:
        with open(valid_responses_file, "r") as f:
            valid_responses = json.load(f)
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading files: {e}")
        return
    
    total_evaluations = len(valid_responses)
    correct_evals = 0
    metric_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})

    for response, data in zip(valid_responses, dataset):
        transformed_selected = json.loads(data["selected"])
        metrics = {}
        
        for key in data.keys():
            if key.startswith("cf_ild") or key.startswith("cb_ild") or key.startswith("ease_ild") \
               or key.startswith("genres") or key.startswith("tags") or key.startswith("bin_div"):
                metrics[key] = transform_to_indices(json.loads(data[key]))
        
        if response["output"] == transformed_selected:
            correct_evals += 1
        
        for key, transformed_metric in metrics.items():
            metric_accuracies[key]["total"] += 1
            if compare_with_x_ignore(response["output"], transformed_metric):
                metric_accuracies[key]["correct"] += 1
    
    accuracy = (correct_evals / total_evaluations * 100) if total_evaluations > 0 else 0
    metric_accuracy_results = {k: (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0 for k, v in metric_accuracies.items()}
    
    summary_data = {
        "name": EVAL_NAME,
        "total_evaluations": total_evaluations,
        "llm_output": {
            "correct_evaluations": correct_evals,
            "accuracy": f"{accuracy:.2f}%",
        },
        "metric_accuracies": {k: f"{v:.2f}%" for k, v in metric_accuracy_results.items()},
    }

    try:
        with open(output_summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary data saved to {output_summary_file}")
    except IOError as e:
        print(f"Error: Could not write to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")

if __name__ == "__main__":
    analyze_data(VALID_RESPONSES_FILE, DATASET_FILE, OUTPUT_SUMMARY_FILE)
