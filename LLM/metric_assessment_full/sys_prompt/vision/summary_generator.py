import json
from collections import defaultdict

EVAL_NAME = "covers_summary"

VALID_RESPONSES_FILE = "valid_responses_" + EVAL_NAME + ".json"
DATASET_FILE = "final_movie_data.json"
OUTPUT_SUMMARY_FILE = "summary_data_" + EVAL_NAME + ".json"


def analyze_data(valid_responses_file, dataset_file, output_summary_file):
    """
    Analyzes data from two JSON files, calculates evaluation metrics,
    and generates summary data in JSON format.

    Args:
        valid_responses_file (str): Path to the valid responses JSON file.
        dataset_file (str): Path to the dataset JSON file.
        output_summary_file (str): Path to the output summary JSON file.
    """

    try:
        with open(valid_responses_file, "r") as f:
            valid_responses = json.load(f)
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    total_evaluations = len(valid_responses)
    correct_evals = {
        "output": 0,
        "cf_ild": 0,
        "cb_ild": 0,
        "bin_div": 0,
    }
    accuracy = {
        "output": 0.0,
        "cf_ild": 0.0,
        "cb_ild": 0.0,
        "bin_div": 0.0,
    }

    selected_list_to_index = {"A": 0, "B": 1, "C": 2}
    list_metrics_value_counts = defaultdict(int)
    dataset_dict = {
        (item["participation"]): item for item in dataset
    }

    for response in valid_responses:
        participation = response["participation"]
        gold = response["gold"]

        dataset_entry = dataset_dict.get(participation)
        if dataset_entry is None:
            print(f"Warning: No corresponding dataset entry found for participation {participation}. Skipping evaluation.")
            total_evaluations -= 1
            continue # skip current evaluation

        cf_ild = dataset_entry.get("cf_ild")
        cb_ild = dataset_entry.get("cb_ild")
        bin_div = dataset_entry.get("bin_div")
        output = response["output"]

        if output and output == gold:
            correct_evals["output"] += 1
        if cf_ild and output == cf_ild:
            correct_evals["cf_ild"] += 1
        if cb_ild and output == cb_ild:
            correct_evals["cb_ild"] += 1
        if bin_div and output == bin_div:
            correct_evals["bin_div"] += 1

        list_metrics = dataset_entry.get("list_metrics")

        if output in selected_list_to_index and list_metrics:
            index = selected_list_to_index[output]
            metric = list_metrics[index]
            list_metrics_value_counts[metric] += 1

    for key in correct_evals:
        accuracy[key] = (
            round(correct_evals[key] / total_evaluations if total_evaluations > 0 else 0.0, 4)
        )

    summary_data = {
        "total_evaluations": total_evaluations,
        "correct_evaluations": correct_evals,
        "correlations": accuracy,
        "chosen_metrics_counts": dict(list_metrics_value_counts),
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