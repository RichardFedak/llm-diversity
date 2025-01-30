import time
import os
import csv
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

step1_sys_prompt = """
You are a thoughtful and knowledgeable assistant tasked with analyzing three lists of movies. For each movie, the title is provided. Use your expertise in film and storytelling to assess the lists and order them from least to most diverse.

Deliver your analysis and the final ordering of the lists in the following JSON format:

{
    "comparison": str,                     # Compare the lists.
    "most_diverse_list_reasoning": str,    # Explanation what list/s can be considered the most diverse.
    "least_diverse_list_reasoning": str,   # Explanation what list/s can be considered the least diverse.
    "final_ordering": array,               # Array, ordering of the lists 'A', 'B' and 'C' starting from least diverse list to most diverse list.
    "second_ordering": array               # Optional array, second possible ordering of the lists if the first ordering is hard to decide. Leave empty if not needed.
}
"""


step1_model = genai.GenerativeModel(system_instruction=step1_sys_prompt, generation_config={"response_mime_type": "application/json"})

total_evaluations = 0
valid_outputs = 0
invalid_outputs = 0
correct_outputs = 0
incorrect_outputs = 0
metric_stats = {}

start_time = time.time()

error_log_file = "invalid_responses_think_single_two_options_titles_no_instructions.log"
valid_responses_file = "valid_responses_think_single_two_options_titles_no_instructions.json"

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open("final_movie_data.json", 'r') as f:
    data = json.load(f)

def convert_to_indices(char_list):
    return [ord(char.upper()) - ord('A') for char in char_list]

with open(valid_responses_file, 'w') as valid_responses_log:
    json_log_data = []  # Collect valid responses to write at the end

    with open(error_log_file, 'w') as error_log:
        error_log.write("Invalid Outputs:\n\n")

        for idx, item in enumerate(data):
            print(idx)
            current_time = time.time()
            if requests_made >= MAX_REQUESTS_PER_MINUTE:
                elapsed_time = current_time - last_request_time
                if elapsed_time < 60:
                    time_to_wait = 80 - elapsed_time
                    print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                    time.sleep(time_to_wait)
                requests_made = 0
                last_request_time = time.time()

            alphas = [float(x.strip()) for x in item['alphas'][1:-1].split(',')]
            if set([0.0, 0.01]).issubset(alphas) or set([1.0, 0.99]).issubset(alphas):
                continue

            list_A = item['list1']
            list_B = item['list2']
            list_C = item['list3']
            ordering = item['selected']
            metric = item['compare_alphas_metric']


            titles_A = {movie['title'] for movie in list_A}
            titles_B = {movie['title'] for movie in list_B}
            titles_C = {movie['title'] for movie in list_C}
            common_titles = titles_A & titles_B & titles_C
            
            filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
            filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
            filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]
            
            list_A_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_A])
            list_B_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_B])
            list_C_info = "\n".join([f"- {movie['title']}" for movie in filtered_list_C])

            prompt = (
                f"List A movies:\n{list_A_info}\n\n"
                f"List B movies:\n{list_B_info}\n\n"
                f"List C movies:\n{list_C_info}"
                )
            gold = [int(x.strip()) for x in ordering[1:-1].split(',')]

            try:
                response_step1 = step1_model.generate_content(prompt)
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
                        if len(output["second_ordering"]) > 0:
                            output["second_ordering"] = convert_to_indices(output["second_ordering"])
                            if output["second_ordering"] == gold:
                                print("ok2")
                                correct_outputs += 1
                                metric_stats[metric].setdefault("correct",0)
                                metric_stats[metric]["correct"] += 1
                            else:
                                incorrect_outputs += 1
                                metric_stats[metric].setdefault("incorrect",0)
                                metric_stats[metric]["incorrect"] += 1
                        else:
                            incorrect_outputs += 1
                            metric_stats[metric].setdefault("incorrect",0)
                            metric_stats[metric]["incorrect"] += 1
                    print(gold)
                    print(output["final_ordering"])
                    print(output["second_ordering"])
                    json_log_data.append({
                        "prompt": prompt,
                        "gold": gold,
                        "output": output,
                        "correct": correctness
                    })
                else:
                    invalid_outputs += 1
                    error_log.write(f"Prompt: {prompt}\nStep 1 Output: {output}\n")
            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                error_log.write(f"Prompt: {prompt}\nError: {str(e)}\n\n")

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

with open("evaluation_summary_think_single_two_options_titles_no_instructions.log", 'w') as summary_log:
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

