import json

input_file = 'final_movie_data.json'
output_file = 'metrics.json'

fields_to_extract = ["cf_ild", "cb_ild", "ease_ild", "genres", "tags", "bin_div"]


with open(input_file, 'r') as infile:
    data = json.load(infile)

def process_field(value):
    values_list = json.loads(value)  # Convert string to list
    if len(values_list) == 3 and len(set(values_list)) < 3:
        return []  # Return empty list if any two values are the same
    return [i[0] for i in sorted(enumerate(values_list), key=lambda x: x[1])]

extracted_data = []
for item in data:
    filtered_item = {}
    alphas = [float(x.strip()) for x in item['alphas'][1:-1].split(',')]
    if set([0.0, 0.01]).issubset(alphas) or set([1.0, 0.99]).issubset(alphas):
        continue
    for key in fields_to_extract:
        if key in item:
            filtered_item[key] = process_field(item.get(key))
    extracted_data.append(filtered_item)

with open(output_file, 'w') as outfile:
    json.dump(extracted_data, outfile, indent=4)

print(f"Filtered data saved successfully to '{output_file}'")

