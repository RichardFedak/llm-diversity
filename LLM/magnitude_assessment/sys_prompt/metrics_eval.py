import json

metrics_file = 'metrics.json'
orderings_file = 'final_orderings_valid_responses_think_single_two_options.json'
output_file = 'statistics_valid_responses_think_single_two_options.json'

fields_to_check = ["cf_ild", "cb_ild", "ease_ild", "genres", "tags", "bin_div"]

with open(metrics_file, 'r') as mfile:
    metrics_data = json.load(mfile)

with open(orderings_file, 'r') as ofile:
    orderings_data = json.load(ofile)

if len(metrics_data) != len(orderings_data):
    print(len(metrics_data), len(orderings_data))
    raise ValueError("Different number of items in metrics and orderings")

statistics = {field: {"correct": 0, "incorrect": 0, "total": 0, "accuracy": 0.0} for field in fields_to_check}

for metric_item, ordering_item in zip(metrics_data, orderings_data):
    for field in fields_to_check:
        metric_value = metric_item.get(field)
        if isinstance(metric_value, list) and metric_value:  # Ignore empty lists
            statistics[field]["total"] += 1
            if metric_value == ordering_item:
                statistics[field]["correct"] += 1
            else:
                statistics[field]["incorrect"] += 1

for field, stats in statistics.items():
    if stats["total"] > 0:
        stats["accuracy"] = stats["correct"] / stats["total"]

with open(output_file, 'w') as outfile:
    json.dump(statistics, outfile, indent=4)

print(f"Statistics saved successfully to '{output_file}'")
