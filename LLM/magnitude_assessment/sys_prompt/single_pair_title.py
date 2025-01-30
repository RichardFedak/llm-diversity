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
You are a thoughtful and knowledgeable assistant tasked with analyzing the diversity in two lists of movies. For each movie, the title is provided. Use your expertise in film and storytelling to assess the uniqueness and variety of the movies within and across the lists.

When analyzing, keep the following in mind:

1. General Approach to Diversity:
   - Evaluate the lists based on how much variety they offer in terms of genres and storytelling styles.
   - Use your understanding of movies and franchises to identify if certain lists feel repetitive or lack diversity due to recurring themes or similar genres. For example, a list dominated by fantasy epics like The Lord of the Rings or Harry Potter might have less diversity compared to a list spanning multiple themes and genres like drama, sci-fi, and thriller.

2. Storytelling, Plot, and Themes:
   - When genre diversity is limited, delve into the storytelling approaches, originality, and thematic depth of the movies.
   - Identify whether the themes and plots reflect a broad spectrum of human experience or feel constrained to specific tropes or archetypes.

Deliver your analysis and and indicate which list is more diverse in each pair in the following JSON format:

{
  "analysis": str            # Compare the lists.
  "more_diverse_list": str,  # The list that is more diverse, 'A' or 'B'.
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

error_log_file = "pairwise_invalid_responses_pairwise_filtered_more_div_think_single_titles.log"
valid_responses_file = "pairwise_valid_responses_pairwise_more_div_think_single_titles.json"

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open("final_movie_data.json", 'r') as f:
    data = json.load(f)

def convert_to_indices(char):
    return ord(char.upper()) - ord('A')

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

            pairs = []

            pairs.append({
                'A': list_A, 
                'B': list_B, 
                'more_diverse': 'A' if ordering.index('0') > ordering.index('1') else 'B'
            })
            pairs.append({
                'A': list_A, 
                'B': list_C, 
                'more_diverse': 'A' if ordering.index('0') > ordering.index('2') else 'B'
            })
            pairs.append({
                'A': list_B, 
                'B': list_C, 
                'more_diverse': 'A' if ordering.index('1') > ordering.index('2') else 'B'
            })

            for pair in pairs:
                list_A = pair['A']
                list_B = pair['B']
                more_diverse = pair['more_diverse']

                list_A_info = "\n".join([f"- {movie['title']}" for movie in list_A])
                list_B_info = "\n".join([f"- {movie['title']}" for movie in list_B])

                prompt = (
                    f"List A:\n{list_A_info}\n\n"
                    f"List B:\n{list_B_info}"
                    )
                gold = more_diverse

                try:
                    response_step1 = step1_model.generate_content(prompt)
                    output = json.loads(response_step1.text.strip())

                    if all(key in output for key in ["analysis", "more_diverse_list"]):

                        valid_outputs += 1
                        metric_stats.setdefault(metric,{})
                        correctness = output["more_diverse_list"] == gold
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
                            "output": output,
                            "correct": correctness
                        })
                    else:
                        invalid_outputs += 1
                        error_log.write(f"Prompt: {prompt}\nStep 1 Output: {output}\n")
                except Exception as e:
                    print(f"Error generating response for row {idx}: {e}")

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

with open("pairwise_more_div_think_single_titles.log", 'w') as summary_log:
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

