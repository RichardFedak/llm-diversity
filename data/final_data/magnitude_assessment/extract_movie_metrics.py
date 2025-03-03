import json

input_file = 'final_movie_data.json'

fields_to_extract = ["cf_ild", "cb_ild", "ease_ild", "genres", "tags", "bin_div"]

def process_field(value):
    values = json.loads(value)
    unique_values = set(values)
    if len(values) == 3 and len(unique_values) < 3:
        return ["X" if values.count(v) > 1 else i for i, v in enumerate(values)]
    return [i[0] for i in sorted(enumerate(values), key=lambda x: x[1])]

with open(input_file, 'r+') as file:
    data = json.load(file)
    
    for item in data:
        for key in fields_to_extract:
            if key in item:
                item[key] = process_field(item[key])
    
    file.seek(0)
    json.dump(data, file, indent=4)
    file.truncate()
