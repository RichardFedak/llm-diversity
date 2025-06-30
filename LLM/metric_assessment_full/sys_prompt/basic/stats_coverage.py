import os
import json
import itertools

# Constants
POSTFIXES = [
    "genres", "plot", "genres_plot",
    "title", "title_genres", "title_plot", "title_genres_plot"
]
PREFIXES = ["_basic_", "_summary_", "_standouts_"]
RESULTS_DIR = "results"

def load_file_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_coverage(prefix: str, directory: str = RESULTS_DIR):
    file_map = {}
    alignment_map = {}

    for postfix in POSTFIXES:
        file_key = f"{prefix}{postfix}"
        path = os.path.join(directory, file_key + ".json")
        if os.path.isfile(path):
            data = load_file_data(path)
            file_map[file_key] = data["evaluations"]
            alignment_map[file_key] = data["accuracy"]
        else:
            print(f"File not found: {path}")

    pairs = itertools.combinations(file_map.items(), 2)
    coverage_results = {}

    for (name_a, evals_a), (name_b, evals_b) in pairs:
        pair_key = f"A_{name_a}|B_{name_b}"
        both_aligned = 0
        at_least_one_aligned = 0
        only_one_aligned = 0

        for eval_a, eval_b in zip(evals_a, evals_b):
            ca = eval_a["correct"]
            cb = eval_b["correct"]

            if ca or cb:
                at_least_one_aligned += 1
            if ca and cb:
                both_aligned += 1
            if (ca and not cb) or (cb and not ca):
                only_one_aligned += 1

        total = len(evals_a)
        at_least_one_aligned_pct = at_least_one_aligned / total if total else 0
        both_aligned_pct = both_aligned / total if total else 0
        only_one_aligned_pct = only_one_aligned / total if total else 0
        both_within_at_least_one_pct = both_aligned / at_least_one_aligned if at_least_one_aligned else 0
        only_one_within_at_least_one_pct = only_one_aligned / at_least_one_aligned if at_least_one_aligned else 0


        coverage_results[pair_key] = {
            "at_least_one_aligned": at_least_one_aligned,
            "at_least_one_aligned_pct": round(at_least_one_aligned_pct, 2),

            "both_aligned": both_aligned,
            "both_aligned_pct": round(both_aligned_pct, 2),
            "both_within_at_least_one_pct": round(both_within_at_least_one_pct, 2),

            "only_one_aligned": only_one_aligned,
            "only_one_aligned_pct_p": round(only_one_aligned_pct, 2),
            "only_one_within_at_least_one_pct": round(only_one_within_at_least_one_pct, 2),

            "A_alignment": round(alignment_map.get(name_a, 0.0), 2),
            "B_alignment": round(alignment_map.get(name_b, 0.0), 2),
        }

    sorted_results = dict(sorted(
        coverage_results.items(),
        key=lambda item: item[1]["at_least_one_aligned_pct"],
        reverse=True
    ))

    return sorted_results

for prefix in PREFIXES:
    print(f"Processing prefix: {prefix}")
    coverage = calculate_coverage(prefix)
    outname = f"coverage_{prefix.strip('_')}.json"
    outpath = os.path.join(RESULTS_DIR, outname)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)
    print(f"Saved to {outpath}")
