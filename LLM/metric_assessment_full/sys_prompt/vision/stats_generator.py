import json
import os
from collections import defaultdict, Counter

files = [
    #"valid_responses_covers_think.json",
    #"valid_responses_covers_think_titles.json",
    "valid_responses_covers_think_genres.json",
    "valid_responses_covers_think_plot.json",
    #"valid_responses_covers_think_full.json",
    "valid_responses_covers_summary.json",
]

def load_data(files):
    data = {}
    for file in files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data[file] = json.load(f)
        else:
            print(f"Warning: {file} not found!")
    return data

def analyze_correctness(data):
    participation_stats = defaultdict(lambda: {'correct_files': 0, 'total_files': 0, 'files': [], 'outputs': []})
    overall_correct = set()
    total_entries = 0
    correct_entries = 0
    
    for file, entries in data.items():
        for entry in entries:
            participation = entry['participation']
            correct = entry['correct']
            output = entry['output']
            gold = entry['gold']
            
            participation_stats[participation]['total_files'] += 1
            participation_stats[participation]['files'].append(file)
            participation_stats[participation]['outputs'].append(output)
            
            if correct:
                participation_stats[participation]['correct_files'] += 1
                overall_correct.add(participation)
                correct_entries += 1
            total_entries += 1
    
    ensemble_correct_count = 0
    for participation, stats in participation_stats.items():
        most_common_output, _ = Counter(stats['outputs']).most_common(1)[0]
        if most_common_output == gold:
            ensemble_correct_count += 1
    
    overall_coverage = len(overall_correct) / len(participation_stats) if participation_stats else 0
    
    return {
        'total_entries': total_entries,
        'correct_entries': correct_entries,
        'overall_correct_count': len(overall_correct),
        'total_participations': len(participation_stats),
        'overall_coverage': overall_coverage,
        'ensemble_correct_count': ensemble_correct_count,
        'participation_stats': dict(participation_stats)
    }

def main():
    data = load_data(files)
    stats = analyze_correctness(data)
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    
if __name__ == "__main__":
    main()
