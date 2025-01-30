import json

file_path = "valid_responses_more_div_think_single.json"
with open(file_path, "r") as file:
    data = json.load(file)

one_correct = 0
none_correct = 0
total = 0

for entry in data:
    gold = entry["gold"]
    final_ordering = entry["output"]["final_ordering"] # or ["output"] for think_single
    
    correct_positions = sum(1 for i in range(len(gold)) if gold[i] == final_ordering[i])
    
    if correct_positions == 1:
        one_correct += 1
    elif correct_positions == 0:
        none_correct += 1
    
    total += 1

one_correct_percent = (one_correct / total) * 100
none_correct_percent = (none_correct / total) * 100

print(f"Wrong orderings: {total}")
print(f"Orderings with one correct position: {one_correct} ({one_correct_percent:.2f}%)")
print(f"Orderings with no correct positions: {none_correct} ({none_correct_percent:.2f}%)")
