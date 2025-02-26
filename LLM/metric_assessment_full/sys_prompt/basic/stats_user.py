import json
from collections import defaultdict


DATASET_FILE = "final_movie_data.json"
OUTPUT_SUMMARY_FILE = "user_metric_stats.json"


def analyze_data(dataset_file, output_summary_file):
    """
    Analyzes data and generates summary for User/metric outputs in JSON format.

    Args:
        dataset_file (str): Path to final movie data file.
        output_summary_file (str): Path to the output summary JSON file.
    """

    try:
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
    
    total_evaluations = len(dataset)
    correct_evals = {"cf_ild": 0, "cb_ild": 0, "bin_div": 0}
    accuracy = {"cf_ild": 0.0, "cb_ild": 0.0, "bin_div": 0.0}

    selected_list_to_index = {"A": 0, "B": 1, "C": 2}
    list_metrics_value_counts = defaultdict(int)

    for response in dataset:

        cf_ild = response.get("cf_ild")
        cb_ild = response.get("cb_ild")
        bin_div = response.get("bin_div")
        output = response.get("selected_list")

        if cf_ild and output == cf_ild:
            correct_evals["cf_ild"] += 1
        if cb_ild and output == cb_ild:
            correct_evals["cb_ild"] += 1
        if bin_div and output == bin_div:
            correct_evals["bin_div"] += 1

        list_metrics = response.get("list_metrics")
        if output in selected_list_to_index and list_metrics:
            index = selected_list_to_index[output]
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
        "name": "user_metric_stats",
        "total_evaluations": total_evaluations,
        "user_output": {
            "correct_evaluations": correct_evals,
            "accuracy": accuracy,
        },
        "chosen_metrics": {
            "counts": dict(list_metrics_value_counts),
            "percentages": chosen_metrics_percentages,
        },
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
    analyze_data(DATASET_FILE, OUTPUT_SUMMARY_FILE)
