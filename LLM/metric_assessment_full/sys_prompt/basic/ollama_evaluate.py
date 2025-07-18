import os
import json
import time
from tqdm import tqdm
from ollama import chat
from pydantic import BaseModel

import subprocess
from dotenv import load_dotenv

load_dotenv()

SSH_CONNECTION = os.getenv("SSH_CONNECTION")

ssh_process = None

def start_ssh_tunnel():
    global ssh_process
    if ssh_process and ssh_process.poll() is None:
        return  # tunnel alive
    print("Starting SSH tunnel ...")
    try:
        ssh_process = subprocess.Popen(SSH_CONNECTION)
        time.sleep(2)
        print("SSH tunnel established.")
    except Exception as exc:
        print(exc)
        ssh_process = None

def restart_ssh_tunnel():
    global ssh_process
    if ssh_process:
        ssh_process.terminate()
        ssh_process.wait()
    start_ssh_tunnel()

start_ssh_tunnel()

class Response(BaseModel):
    list_A_description: str
    list_B_description: str
    list_C_description: str
    comparison: str
    most_diverse_list_reasoning: str
    most_diverse_list: str

calculated_results_folder = 'results'
result_folder = 'results_ollama'

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
                for attempt in range(2):
                    try:
                        response = chat(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama3.1:8b",
                            format=Response.model_json_schema(),
                        )
                        break
                    except Exception as e:
                        print(f"Chat failed on attempt {attempt+1}: {e}")
                        if attempt == 0:
                            restart_ssh_tunnel()
                        else:
                            raise
                response_data = Response.model_validate_json(response.message.content)
                most_diverse_list = response_data.most_diverse_list
                correct = most_diverse_list == gold
                correct_outputs += 1 if correct else 0
                evaluations.append({
                    'participation': participation,
                    'prompt': prompt,
                    'response': response_data.model_dump(),
                    'gold': gold,
                    'output': most_diverse_list,
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

if ssh_process:
    ssh_process.terminate()