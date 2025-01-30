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
You are a helpful assistant. Your goal is to analyze the diversity in three lists of movies. For each movie, the title, genres, and plot are provided.

Focus on the following two aspects:
1. Genres - This is the most important factor in determining diversity.
2. Plot and Themes - Only if the genres are almost the same or overlap significantly between the lists, use plot and themes to differentiate them.

1. Focus on Genres:
   - Assess the variety and uniqueness of genres in each list.
   - Identify how diverse the genres are within each list and across lists.

2. Focus on Plot and Themes:
   - If genres are similar or overlap significantly between the lists:
    - Analyze the plots to find nuanced differences in storytelling approaches.
    - Evaluate the overarching themes and how they contribute to the distinctiveness of each list.

Provide your response in the following JSON schema:

{
  "list_a_analysis": str,  # Describe the diversity in list A.
  "list_b_analysis": str,  # Describe the diversity in list B.
  "list_c_analysis": str,  # Describe the diversity in list C.
  "comparison": str        # Compare the lists based on their variety in genres and movie themes.
}
"""

step2_sys_prompt = """
You are an evaluator given an analysis of three movie lists. Your task is to order the lists from least to most diverse based on the provided analysis.
Even if the lists are very similar or same you MUST keep the output format.

Use the following JSON schema for your response:
{
  "reasoning": str,         # In short, explain your evaluation of the analysis and your decision.
  "final_ordering": array   # Array, ordering of the lists 'A', 'B' and 'C' starting from least diverse list.
}
"""

step1_model = genai.GenerativeModel(system_instruction=step1_sys_prompt, generation_config={"response_mime_type": "application/json"})
step2_model = genai.GenerativeModel(system_instruction=step2_sys_prompt, generation_config={"response_mime_type": "application/json"})

total_evaluations = 0
valid_outputs = 0
invalid_outputs = 0
correct_outputs = 0
incorrect_outputs = 0
metric_stats = {}

start_time = time.time()

error_log_file = "invalid_responses_filtered.log"
valid_responses_file = "valid_responses.json"

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
            if idx > 2:
                continue
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

            list_A = item['list1']
            list_B = item['list2']
            list_C = item['list3']
            ordering = item['selected']
            metric = item['selected_metric']

            titles_A = {movie['title'] for movie in list_A}
            titles_B = {movie['title'] for movie in list_B}
            titles_C = {movie['title'] for movie in list_C}
            common_titles = titles_A & titles_B & titles_C
            
            filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
            filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
            filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]
            
            list_A_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_A])
            list_B_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_B])
            list_C_info = "\n".join([f"- {movie['title']} - Genres of the movie: {movie['genres']} - Plot of the movie: {movie['plot']}" for movie in filtered_list_C])

            prompt = (
                f"List A:\n{list_A_info}\n\n"
                f"List B:\n{list_B_info}\n\n"
                f"List C:\n{list_C_info}"
                )
            gold = [int(x.strip()) for x in ordering[1:-1].split(',')]

            try:
                response_step1 = step1_model.generate_content(prompt)
                step1_output = json.loads(response_step1.text.strip())
                time.sleep(REQUEST_INTERVAL)

                if all(key in step1_output for key in ["list_a_analysis", "list_b_analysis", "list_c_analysis", "comparison"]):
                    step2_input = json.dumps(step1_output)
                    response_step2 = step2_model.generate_content(step2_input)
                    step2_output = json.loads(response_step2.text.strip())

                    if step2_output["final_ordering"]:
                        step2_output["final_ordering"] = convert_to_indices(step2_output["final_ordering"])
                        valid_outputs += 1
                        metric_stats.setdefault(metric,{})
                        correctness = step2_output["final_ordering"] == gold
                        if correctness:
                            correct_outputs += 1
                            metric_stats[metric].setdefault("correct",0)
                            metric_stats[metric]["correct"] += 1
                        else:
                            incorrect_outputs += 1
                            metric_stats[metric].setdefault("incorrect",0)
                            metric_stats[metric]["incorrect"] += 1

                            json_log_data.append({
                                "prompt": prompt,
                                "gold": gold,
                                "step1_output": step1_output if idx < 10 else "",
                                "step2_output": step2_output,
                                "correct": correctness
                            })
                    else:
                        invalid_outputs += 1
                        error_log.write(f"Prompt: {prompt}\nStep 1 Output: {step1_output}\nInvalid Step 2 Output: {step2_output}\n\n")
                else:
                    invalid_outputs += 1
                    error_log.write(f"Prompt: {prompt}\nInvalid Step 1 Output: {step1_output}\n\n")
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

with open("evaluation_summary_filtered.log", 'w') as summary_log:
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

print(f"\nDetailed invalid outputs logged to: {error_log_file}")
print(f"Summary of evaluation logged to: evaluation_summary_filtered.log")
print(f"Valid responses logged to: {valid_responses_file}")
