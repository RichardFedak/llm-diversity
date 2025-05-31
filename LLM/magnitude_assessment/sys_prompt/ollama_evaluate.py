import os
import json
import time
from tqdm import tqdm
from ollama import chat
from pydantic import BaseModel

class DiversityScores(BaseModel):
    A: int
    B: int
    C: int

class Response(BaseModel):
    list_A_description: str
    list_B_description: str
    list_C_description: str
    comparison: str
    diversity_scores: DiversityScores

calculated_results_folder = 'results'
result_folder = 'results_ollama'

def convert_to_indices(char_list):
    return [ord(char.upper()) - ord('A') for char in char_list]

os.makedirs(result_folder, exist_ok=True)

for filename in tqdm(os.listdir(calculated_results_folder), desc="Processing files"):
    if filename.endswith('.json'):
        filepath = os.path.join(calculated_results_folder, filename)
        name = filename.replace('.json', '')
        output_filename = f"{name}.json"
        output_filepath = os.path.join(result_folder, output_filename)
        if os.path.exists(output_filepath):
            print(f"Skipping {output_filename}.")
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            start_time = time.time()
            data = json.load(f)
            name = data.get('name')
            system_prompt = data.get('system_prompt')
            evaluations = []
            correct_outputs = 0

            for eval in tqdm(data.get('evaluations', []), desc=f"Evaluating {filename}", leave=False):
                prompt = eval.get('prompt')
                participation = eval.get('participation')
                gold = eval.get('gold')
                response = chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3.1:8b",
                    format=Response.model_json_schema(),
                )
                response_data = Response.model_validate_json(response.message.content)
                scores_dict = response_data.diversity_scores.model_dump()
                final_ordering = convert_to_indices(sorted(scores_dict, key=scores_dict.get))
                correct = final_ordering == gold
                correct_outputs += 1 if correct else 0

                fixed_order = ["A", "B", "C"]
                approx_scores = [round((scores_dict[key] / 10), 2) for key in fixed_order]

                evaluations.append({
                    'participation': participation,
                    'prompt': prompt,
                    'response': response_data.model_dump(),
                    'gold': gold,
                    'final_ordering': final_ordering,
                    'approx_scores': approx_scores,
                    'correct': correct,
                })
            
            elapsed_time = time.time() - start_time
            accuracy_percentage = round((correct_outputs / len(evaluations)) * 100, 2) if evaluations else 0
            with open(output_filepath, 'w', encoding='utf-8') as output_file:
                json.dump({
                    'name': name,
                    'evaluation_duration': elapsed_time,
                    'accuracy': accuracy_percentage,
                    'system_prompt': system_prompt,
                    'evaluations': evaluations
                }, output_file, indent=4)
