import json
from diversity_stimulus import DiversityStimulusAnalyzer, StimulusType, DiversityStimulusEvaluator
from model import *


log_file_path = "evaluation_results.json"

evaluation_data = []

def add_evaluation_data(participation, stimuli_scores, diversity_score, user_answer):
    stimuli_scores_str = {stimulus.value: score for stimulus, score in stimuli_scores.items()}

    log_data = {
        "participation": participation,
        "stimuli_scores": stimuli_scores_str,
        "diversity_score": diversity_score,
        "user_answer": user_answer,
    }

    evaluation_data.append(log_data)

def save_evaluation_data():
    with open(log_file_path, "w") as log_file:
        json.dump(evaluation_data, log_file, indent=4)

with open("data_first5.json", "r") as f:
    data = Model.model_validate(json.load(f))

char_to_digit = {
    'A': 1,
    'B': 2,
    'C': 3,
}

for evaluation in data.root:
    analyzer = DiversityStimulusAnalyzer()

    lists = [evaluation.metric_data.list_A, evaluation.metric_data.list_B, evaluation.metric_data.list_C]
    analyzer.analyze(lists, perceived_most_diverse_index=char_to_digit[evaluation.metric_data.selected_list])

    lists = [evaluation.magnitude_data[0].list1, evaluation.magnitude_data[0].list2, evaluation.magnitude_data[0].list3]
    weights = [float(x.strip()) for x in evaluation.magnitude_data[0].approx_alphas[1:-1].split(',')]
    analyzer.analyze(lists, weights=weights)

    lists = [evaluation.magnitude_data[1].list1, evaluation.magnitude_data[1].list2, evaluation.magnitude_data[1].list3]
    weights = [float(x.strip()) for x in evaluation.magnitude_data[1].approx_alphas[1:-1].split(',')]
    analyzer.analyze(lists, weights=weights)

    stimuli_scores = analyzer.calculate_total_distribution()
    print(stimuli_scores)

    evaluator = DiversityStimulusEvaluator(stimuli_scores)

    for iter in evaluation.recommendation_data.iterations:
        lists = [rec.items for _, rec in iter.iterations.items()]
        diversity_score = evaluator.evaluate_lists(lists)

        print(diversity_score)
        print(iter.diversity_score)

        add_evaluation_data(
            evaluation.participation,
            stimuli_scores,
            diversity_score,
            iter.diversity_score
        )

save_evaluation_data()
