import time
import os
import csv
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

sys_prompt = """
You are an assistant tasked with comparing three lists of movies and assessing their diversity. For each list, movie titles are provided along with their respective genres and plot.
Use your own judgment to determine what information is relevant when assessing the diversity of the lists. You may consider the movie titles, genres and plots provided, but the main source of decision-making should be your own analysis.

Deliver your comparison and choice of the most diverse list in the following JSON format with fields:

{
    "list_A_description": str,             # Based on the movies in the list A, provide a description of the movies in the list. Try to create groups of similar movies, the more groups, the more is the list diverse.
    "list_B_description": str,             # Based on the movies in the list B, provide a description of the movies in the list. Try to create groups of similar movies, the more groups, the more is the list diverse.
    "list_C_description": str,             # Based on the movies in the list C, provide a description of the movies in the list. Try to create groups of similar movies, the more groups, the more is the list diverse.
    "comparison": object,                  # Compare the lists based on the description you created.
    "most_diverse_list_reasoning": str,    # Explanation of what list you perceive to be the most diverse.
    "least_diverse_list_reasoning": str,   # Explanation of what list you perceive to be the least diverse.
    "final_ordering": list                 # The lists ordered from least to most diverse, e.g., ['A', 'B', 'C'] if list A is the least diverse and list C is the most diverse.
}
"""

model = genai.GenerativeModel(system_instruction=sys_prompt, generation_config={"response_mime_type": "application/json"})

total_evaluations = 0
valid_outputs = 0
invalid_outputs = 0
correct_outputs = 0
incorrect_outputs = 0
metric_stats = {}

start_time = time.time()

EVALUATION_NAME = "single_think_custom_plot"

error_log_file = "invalid_responses_"+EVALUATION_NAME+".log"
valid_responses_file = "valid_responses_"+EVALUATION_NAME+".json"
eval_file = "evaluation_summary_"+EVALUATION_NAME+".log"

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open("final_movie_data_with_summary_full.json", 'r') as f:
    data = json.load(f)

def convert_to_indices(char_list):
    return [ord(char.upper()) - ord('A') for char in char_list]

def format_list_info(list_data, list_name):
    movies_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in list_data['items']])
    summary = list_data['summary']


    return f"{list_name}:\n{movies_info}\n\n"

with open(valid_responses_file, 'w') as valid_responses_log:
    json_log_data = []

    with open(error_log_file, 'w') as error_log:
        error_log.write("Invalid Outputs:\n\n")

        for idx, item in enumerate(data):
            print(idx)
            if idx >10:
                break
            current_time = time.time()
            if requests_made >= MAX_REQUESTS_PER_MINUTE:
                elapsed_time = current_time - last_request_time
                if elapsed_time < 60:
                    time_to_wait = 80 - elapsed_time
                    print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                    time.sleep(time_to_wait)
                requests_made = 0
                last_request_time = time.time()

            list_A = item['list1']
            list_B = item['list2']
            list_C = item['list3']
            ordering = item['selected']
            metric = item['compare_alphas_metric']
            gold = [int(x.strip()) for x in ordering[1:-1].split(',')]
            participation = item['participation']

            list_A_info = format_list_info(list_A, "List A")
            list_B_info = format_list_info(list_B, "List B")
            list_C_info = format_list_info(list_C, "List C")

            prompt = f"{list_A_info}\n\n{list_B_info}\n\n{list_C_info}"

            try:
                response_step1 = model.generate_content(prompt)
                output = json.loads(response_step1.text.strip())

                if all(key in output for key in ["most_diverse_list_reasoning", "least_diverse_list_reasoning", "comparison", "final_ordering"]):
                    output["final_ordering"] = convert_to_indices(output["final_ordering"])
                    valid_outputs += 1
                    metric_stats.setdefault(metric,{})
                    correctness = output["final_ordering"] == gold
                    if correctness:
                        print("ok")
                        correct_outputs += 1
                        metric_stats[metric].setdefault("correct",0)
                        metric_stats[metric]["correct"] += 1
                    else:
                        incorrect_outputs += 1
                        metric_stats[metric].setdefault("incorrect",0)
                        metric_stats[metric]["incorrect"] += 1

                    json_log_data.append({
                        "output_full": output,
                        "participation": participation,
                        "gold": gold,
                        "output": output["final_ordering"],
                        "correct": correctness,
                        "comparison": output["comparison"],
                        "most_diverse_list_reasoning": output["most_diverse_list_reasoning"],
                        "least_diverse_list_reasoning": output["least_diverse_list_reasoning"]
                    })
                else:
                    invalid_outputs += 1
                    error_log.write(f"Prompt: {prompt}\nStep 1 Output: {output}\n")
                    json_log_data.append({
                        "participation": participation,
                        "gold": gold,
                        "output": output["final_ordering"],
                        "correct": correctness,
                        "comparison": "X",
                        "most_diverse_list_reasoning": "X",
                        "least_diverse_list_reasoning": "X"
                    })
            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                error_log.write(f"Prompt: {prompt}\nError: {str(e)}\n\n")
                json_log_data.append({
                    "participation": participation,
                    "gold": gold,
                    "output": output["final_ordering"],
                    "correct": correctness,
                    "comparison": "X",
                    "most_diverse_list_reasoning": "X",
                    "least_diverse_list_reasoning": "X"
                })

            total_evaluations += 1
            requests_made += 1
            time.sleep(REQUEST_INTERVAL)

    json.dump(json_log_data, valid_responses_log, indent=4)

valid_percentage = (valid_outputs / total_evaluations) * 100 if total_evaluations > 0 else 0
invalid_percentage = (invalid_outputs / total_evaluations) * 100 if total_evaluations > 0 else 0
accuracy_percentage = (correct_outputs / valid_outputs) * 100 if valid_outputs > 0 else 0

end_time = time.time()
elapsed_time = end_time - start_time

print("\n--- Evaluation Results ---")
print(f"Total evaluations: {total_evaluations}")
print(f"Valid outputs: {valid_outputs} ({valid_percentage:.2f}%)")
print(f"Invalid outputs: {invalid_outputs} ({invalid_percentage:.2f}%)")
print(f"Correct outputs: {correct_outputs}")
print(f"Incorrect outputs: {incorrect_outputs}")
print(f"Accuracy: {accuracy_percentage:.2f}%")
for metric_name, stats in metric_stats.items():
    print(f"  Metric: {metric_name}")
    for stat_name, count in stats.items():
        print(f"    {stat_name}: {count}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

with open(eval_file, 'w') as summary_log:
    summary_log.write("--- Evaluation Results ---\n")
    summary_log.write(f"Total evaluations: {total_evaluations}\n")
    summary_log.write(f"Valid outputs: {valid_outputs} ({valid_percentage:.2f}%)\n")
    summary_log.write(f"Invalid outputs: {invalid_outputs} ({invalid_percentage:.2f}%)\n")
    summary_log.write(f"Correct outputs: {correct_outputs}\n")
    summary_log.write(f"Incorrect outputs: {incorrect_outputs}\n")
    summary_log.write(f"Accuracy: {accuracy_percentage:.2f}%\n")
    for metric_name, stats in metric_stats.items():
        summary_log.write(f"Metric: {metric_name}\n")
        for stat_name, count in stats.items():
            summary_log.write(f"  {stat_name}: {count}\n")
    summary_log.write(f"Total elapsed time: {elapsed_time:.2f} seconds\n")

