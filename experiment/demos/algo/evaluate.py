import json
from diversity_stimulus_analyzer import DiversityStimulusAnalyzer
from model import *


with open("data_first5.json", "r") as f:
    data = Model.model_validate(json.load(f))

char_to_digit = {
    'A': 1,
    'B': 2,
    'C': 3,
}

for evaluation in data.root:
    analyzer = DiversityStimulusAnalyzer()

    # Metric assessment
    lists = []
    lists.append(evaluation.metric_data.list_A)
    lists.append(evaluation.metric_data.list_B)
    lists.append(evaluation.metric_data.list_C)
    analyzer.analyze(lists, perceived_most_diverse_index=char_to_digit[evaluation.metric_data.selected_list])

    # Magnitude assessment
    lists = []
    lists.append(evaluation.magnitude_data[0].list1)
    lists.append(evaluation.magnitude_data[0].list2)
    lists.append(evaluation.magnitude_data[0].list3)

    weights = [float(x.strip()) for x in evaluation.magnitude_data[0].approx_alphas[1:-1].split(',')]
    analyzer.analyze(lists, weights=weights)

    lists = []
    lists.append(evaluation.magnitude_data[1].list1)
    lists.append(evaluation.magnitude_data[1].list2)
    lists.append(evaluation.magnitude_data[1].list3)

    weights = [float(x.strip()) for x in evaluation.magnitude_data[1].approx_alphas[1:-1].split(',')]
    analyzer.analyze(lists, weights=weights)

    final_dist = analyzer.calculate_total_distribution()

    print(final_dist)