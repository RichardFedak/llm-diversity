import os
import csv
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configure model
model_ids = [
    "tunedModels/movieplottrainsplit1-z8pegnfxdnub",
    "tunedModels/movieplottrainsplit2-5quwvzsfa1sp",
    "tunedModels/movieplottrainsplit3-ul02m7s9196n",
    "tunedModels/movieplottrainsplit4-w32s1v9hu",
    "tunedModels/movieplottrainsplit5-kli7uscfxbf0",
]
models = [genai.GenerativeModel(id) for id in model_ids]

# Test files
test_files = [f"test_split_{i}.csv" for i in range(1, 6)]

# Evaluation metrics
total_evaluations = 0
valid_outputs = 0
invalid_outputs = 0
correct_outputs = 0
incorrect_outputs = 0

# Start time
start_time = time.time()

# Log file for invalid responses
error_log_file = "invalid_responses.log"

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 12
REQUEST_INTERVAL = (60 / MAX_REQUESTS_PER_MINUTE) / 2
requests_made = 0  # Track the number of requests made in the current minute
last_request_time = time.time()

with open(error_log_file, 'w') as error_log:
    error_log.write("Invalid Outputs:\n\n")

    # Evaluate test files
    for i in range(len(test_files)):
        test_file = test_files[i]
        print(f"Evaluating test file: {test_file}")

        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader, start=1):
                print(idx)
                # Enforce rate limit
                current_time = time.time()
                if requests_made >= MAX_REQUESTS_PER_MINUTE:
                    elapsed_time = current_time - last_request_time
                    if elapsed_time < 60:
                        time_to_wait = 80 - elapsed_time
                        print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                        time.sleep(time_to_wait)
                    # Reset request counter after waiting
                    requests_made = 0
                    last_request_time = time.time()

                # Process the request
                prompt = row["input"]
                gold = row["output"]

                try:
                    # Generate response
                    response = models[i].generate_content(prompt)
                    generated_output = response.text.strip()

                    if generated_output in {"A", "B"}:
                        valid_outputs += 1
                        if generated_output == gold:
                            correct_outputs += 1
                        else:
                            incorrect_outputs += 1
                    else:
                        invalid_outputs += 1
                        error_log.write(f"Prompt: {prompt}\nExpected: {gold}\nReceived: {generated_output}\n\n")
                except Exception as e:
                    print(f"Error generating response for row {idx}: {e}")
                    invalid_outputs += 1
                    error_log.write(f"Prompt: {prompt}\nError: {str(e)}\n\n")

                total_evaluations += 1
                requests_made += 1
                time.sleep(REQUEST_INTERVAL)  # Respect interval between requests

# Calculate percentages
valid_percentage = (valid_outputs / total_evaluations) * 100 if total_evaluations > 0 else 0
invalid_percentage = (invalid_outputs / total_evaluations) * 100 if total_evaluations > 0 else 0
accuracy_percentage = (correct_outputs / valid_outputs) * 100 if valid_outputs > 0 else 0

# End time
end_time = time.time()
elapsed_time = end_time - start_time

# Print results
print("\n--- Evaluation Results ---")
print(f"Total evaluations: {total_evaluations}")
print(f"Valid outputs: {valid_outputs} ({valid_percentage:.2f}%)")
print(f"Invalid outputs: {invalid_outputs} ({invalid_percentage:.2f}%)")
print(f"Correct outputs: {correct_outputs}")
print(f"Incorrect outputs: {incorrect_outputs}")
print(f"Accuracy (correct/valid): {accuracy_percentage:.2f}%")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

# Log results
with open("evaluation_summary.log", 'w') as summary_log:
    summary_log.write("--- Evaluation Results ---\n")
    summary_log.write(f"Total evaluations: {total_evaluations}\n")
    summary_log.write(f"Valid outputs: {valid_outputs} ({valid_percentage:.2f}%)\n")
    summary_log.write(f"Invalid outputs: {invalid_outputs} ({invalid_percentage:.2f}%)\n")
    summary_log.write(f"Correct outputs: {correct_outputs}\n")
    summary_log.write(f"Incorrect outputs: {incorrect_outputs}\n")
    summary_log.write(f"Accuracy (correct/valid): {accuracy_percentage:.2f}%\n")
    summary_log.write(f"Total elapsed time: {elapsed_time:.2f} seconds\n")

print(f"\nDetailed invalid outputs logged to: {error_log_file}")
print(f"Summary of evaluation logged to: evaluation_summary.log")
