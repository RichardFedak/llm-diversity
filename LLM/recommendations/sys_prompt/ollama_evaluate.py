import os
import json
import time
from tqdm import tqdm
from ollama import chat
from pydantic import BaseModel

class Response(BaseModel):
    list_A_description: str
    list_B_description: str
    list_C_description: str
    list_D_description: str
    list_E_description: str
    list_F_description: str
    diversity_summarization: str
    answer: int

calculated_results_folder = 'diversity_results'
result_folder = 'diversity_results_ollama'

os.makedirs(result_folder, exist_ok=True)

def is_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

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
                block = eval.get('block')
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
                answer_score = response_data.answer
                correct = False
                if is_int(answer_score):
                    answer_score_num = int(answer_score)
                    correct = answer_score_num == int(gold)
                else:
                    correct = False
                correct_outputs += 1 if correct else 0
                evaluations.append({
                    'participation': participation,
                    'block': block,
                    'prompt': prompt,
                    'response': response_data.model_dump(),
                    'gold': gold,
                    'diversity_score': answer_score,
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
