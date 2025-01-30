import json

input_json_path = 'valid_responses_think_single_two_options.json'
output_json_path = 'final_orderings_valid_responses_think_single_two_options.json'

with open(input_json_path, 'r') as infile:
    data = json.load(infile)

final_orderings = [item['output']['final_ordering'] for item in data]

with open(output_json_path, 'w') as outfile:
    json.dump(final_orderings, outfile, indent=4)

print(f"Extracted {len(final_orderings)} orderings, saved to {output_json_path}")
