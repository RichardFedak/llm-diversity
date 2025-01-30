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
You are a helpful assistant. Analyze the diversity in two lists of movies, each containing movie names and genres. 
Evaluate the diversity in each list and compare them based on their variety in genres and movie themes. 

Provide your response in the following JSON schema:
{
  "list_a_analysis": str,  # Describe the diversity in list A.
  "list_b_analysis": str,  # Describe the diversity in list B.
  "comparison": str        # Compare the two lists based on their variety in genres and movie themes.
}
"""

step2_sys_prompt = """
You are an evaluator given an analysis of two movie lists. Your task is to determine which list is more diverse based on the provided analysis. Even if the lists are very similar you MUST choose one that is more diverse.

Use the following JSON schema for your response:
{
  "reasoning": str,         # In short explain your evaluation of the analysis and your decision.
  "more_diverse_list": str  # The list you determine to be more diverse, either 'A' or 'B'.
}
"""

step1_model = genai.GenerativeModel(system_instruction=step1_sys_prompt, generation_config={"response_mime_type": "application/json"})
step2_model = genai.GenerativeModel(system_instruction=step2_sys_prompt, generation_config={"response_mime_type": "application/json"})

total_evaluations = 0
valid_outputs = 0
invalid_outputs = 0
correct_outputs = 0
incorrect_outputs = 0

start_time = time.time()

error_log_file = "invalid_responses_filtered.log"
valid_responses_file = "valid_responses.json"

MAX_REQUESTS_PER_MINUTE = 14
REQUEST_INTERVAL = (80 / MAX_REQUESTS_PER_MINUTE)
requests_made = 0  
last_request_time = time.time()

with open(valid_responses_file, 'w') as valid_responses_log:
    json_log_data = []  # Collect valid responses to write at the end

    with open(error_log_file, 'w') as error_log:
        error_log.write("Invalid Outputs:\n\n")

        with open("data.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader, start=1):
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

                prompt = row["input"]
                gold = row["output"]

                try:
                    # Step 1: Analyze lists
                    response_step1 = step1_model.generate_content(prompt)
                    step1_output = json.loads(response_step1.text.strip())
                    time.sleep(REQUEST_INTERVAL)

                    # Validate Step 1 output structure
                    if all(key in step1_output for key in ["list_a_analysis", "list_b_analysis", "comparison"]):
                        # Step 2: Evaluate and decide
                        step2_input = json.dumps(step1_output)
                        response_step2 = step2_model.generate_content(step2_input)
                        step2_output = json.loads(response_step2.text.strip())

                        # Validate Step 2 output
                        if step2_output["more_diverse_list"] in {"A", "B"}:
                            valid_outputs += 1
                            correctness = step2_output["more_diverse_list"] == gold
                            if correctness:
                                correct_outputs += 1
                            else:
                                incorrect_outputs += 1

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
print(f"Accuracy (correct/valid): {accuracy_percentage:.2f}%")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

with open("evaluation_summary_filtered.log", 'w') as summary_log:
    summary_log.write("--- Evaluation Results ---\n")
    summary_log.write(f"Total evaluations: {total_evaluations}\n")
    summary_log.write(f"Valid outputs: {valid_outputs} ({valid_percentage:.2f}%)\n")
    summary_log.write(f"Invalid outputs: {invalid_outputs} ({invalid_percentage:.2f}%)\n")
    summary_log.write(f"Correct outputs: {correct_outputs}\n")
    summary_log.write(f"Incorrect outputs: {incorrect_outputs}\n")
    summary_log.write(f"Accuracy (correct/valid): {accuracy_percentage:.2f}%\n")
    summary_log.write(f"Total elapsed time: {elapsed_time:.2f} seconds\n")

print(f"\nDetailed invalid outputs logged to: {error_log_file}")
print(f"Summary of evaluation logged to: evaluation_summary_filtered.log")
print(f"Valid responses logged to: {valid_responses_file}")
