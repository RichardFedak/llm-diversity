import os
import time
import json
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

sys_prompt = """
You are an assistant tasked with comparing three lists of movies. For each movie, the cover is provided. Use your expertise to assess the lists and choose the one with the most diverse collection of movies based solely on the information that can be obtained from the covers.

Deliver your comparison and choice of the most diverse list in the following JSON format:

{
    "comparison": str,                     # Compare the lists.
    "most_diverse_list_reasoning": str,    # Explanation of what list you perceive to be the most diverse.
    "most_diverse_list": str               # The list you determine to be the most diverse, either 'A', 'B' or 'C'.
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

EVALUATION_NAME = "covers_think"

error_log_file = "invalid_responses_"+EVALUATION_NAME+".log"
valid_responses_file = "valid_responses_"+EVALUATION_NAME+".json"
eval_file = "evaluation_summary_"+EVALUATION_NAME+".log"

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open("final_movie_data.json", 'r') as f:
    data = json.load(f)

def fetch_save_and_upload_image(url, local_path):
    response = httpx.get(url)
    
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        uploaded_file = genai.upload_file(local_path)
        os.remove(local_path)
        
        return uploaded_file
    else:
        print(f"Failed to retrieve image from {url}")
        return None
    
def process_image_list(covers, list_name):
    uploaded_images = []
    for idx, url in enumerate(covers):
        local_path = f"{list_name}_{idx}.jpg"  # Local file name for the image
        uploaded_uri = fetch_save_and_upload_image(url, local_path)
        if uploaded_uri:
            uploaded_images.append(uploaded_uri)
    return uploaded_images

with open(valid_responses_file, 'w') as valid_responses_log:
    json_log_data = []

    with open(error_log_file, 'w') as error_log:
        error_log.write("Invalid Outputs:\n\n")

        for idx, item in enumerate(data):
            print(idx)
            if idx > 5:
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

            list_A = item['list_A']
            list_B = item['list_B']
            list_C = item['list_C']
            gold_most_diverse = item['selected_list']
            participation = item['participation']

            titles_A = {movie['title'] for movie in list_A}
            titles_B = {movie['title'] for movie in list_B}
            titles_C = {movie['title'] for movie in list_C}
            common_titles = titles_A & titles_B & titles_C
            
            filtered_list_A = [movie for movie in list_A if movie['title'] not in common_titles]
            filtered_list_B = [movie for movie in list_B if movie['title'] not in common_titles]
            filtered_list_C = [movie for movie in list_C if movie['title'] not in common_titles]
            
            list_A_covers = [movie['cover'] for movie in filtered_list_A]
            list_B_covers = [movie['cover'] for movie in filtered_list_B]
            list_C_covers = [movie['cover'] for movie in filtered_list_C]

            try:
                uploaded_list1 = process_image_list(list_A_covers, "list1")
                uploaded_list2 = process_image_list(list_B_covers, "list2")
                uploaded_list3 = process_image_list(list_C_covers, "list3")

                prompt =(
                    ["List A:\n"] + uploaded_list1 + ["\n\nList B:\n"] + uploaded_list2 + ["\n\nList C:\n"] + uploaded_list3
                )
                response_step1 = model.generate_content(prompt)
                output = json.loads(response_step1.text.strip())

                if all(key in output for key in ["comparison", "most_diverse_list_reasoning", "most_diverse_list"]):

                    valid_outputs += 1
                    metric_stats.setdefault("cf_ild",{})
                    metric_stats.setdefault("cb_ild",{})
                    metric_stats.setdefault("bin_div",{})

                    correctness = output["most_diverse_list"] == gold_most_diverse
                    if correctness:
                        print("ok")
                        correct_outputs += 1
                    else:
                        incorrect_outputs += 1

                    for metric in ["cf_ild", "cb_ild", "bin_div"]:
                        metric_correctness = output["most_diverse_list"] == item[metric]
                        if metric_correctness:
                            metric_stats[metric].setdefault("correct", 0)
                            metric_stats[metric]["correct"] += 1
                        else:
                            metric_stats[metric].setdefault("incorrect", 0)
                            metric_stats[metric]["incorrect"] += 1

                    json_log_data.append({
                        "participation": participation,
                        "comparison": output["comparison"],
                        "most_diverse_list_reasoning": output["most_diverse_list_reasoning"],
                        "gold": gold_most_diverse,
                        "output": output["most_diverse_list"],
                        "correct": correctness
                    })
                else:
                    invalid_outputs += 1
                    error_log.write(f"Participation: {participation}\nStep 1 Output: {output}\n")
                    json_log_data.append({
                        "participation": participation,
                        "comparison": "X",
                        "most_diverse_list_reasoning": "X",
                        "gold": gold_most_diverse,
                        "output": "X",
                        "correct": False
                    })
            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                error_log.write(f"Participation: {participation}\nError: {str(e)}\n\n")
                json_log_data.append({
                    "participation": participation,
                    "comparison": "X",
                    "most_diverse_list_reasoning": "X",
                    "gold": gold_most_diverse,
                    "output": "X",
                    "correct": False
                })

            total_evaluations += 1
            requests_made += 1
            # time.sleep(REQUEST_INTERVAL) # uploading images is slow enough (~40s)

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
