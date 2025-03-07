import time
import json
import os
from enum import Enum
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class MovieFields(Enum):
    TITLE = "title"
    PLOT = "plot"
    GENRES = "genres"

class MovieEvaluator:
    def __init__(self, api_key, evaluation_name, system_prompt=None, input_fields=None, include_summary=False, temperature=0):
        """Initializes the MovieEvaluator with API key and configuration."""
        genai.configure(api_key=api_key)
        self.system_prompt = system_prompt
        self.input_fields = input_fields
        self.evaluation_name = evaluation_name
        self.include_summary = include_summary
        self.temperature = temperature
        self.MAX_REQUESTS_PER_MINUTE = 15
        self.REQUEST_INTERVAL = 60 / self.MAX_REQUESTS_PER_MINUTE
        self.results = []

    def _generate_list_info(self, list_data):
        list_info = []
        for movie in list_data:
            movie_info = []
            for field in self.input_fields:
                value = movie[field.value]
                movie_info.append(f"{field.name}: {value}")
            list_info.append("\n".join(movie_info) + "\n")
        return "\n".join(list_info)

    def _generate_summary_text(self, summary):
        summary_text = (
            f"Popularity Diversity: {summary['popularity_diversity']['reasoning']}\n"
            f"Genre Diversity: {summary['genre_diversity']['reasoning']}\n"
            f"Theme Diversity: {summary['theme_diversity']['reasoning']}\n"
            f"Time Span: {summary['time_span']['reasoning']}\n"
            f"Franchise Inclusion: {'Yes' if summary['franchise_inclusion']['value'] else 'No'} - {summary['franchise_inclusion']['reasoning']}"
        )
        return summary_text

    def _load_existing_results(self, json_log_file):
        """Loads existing evaluation results if the file exists."""
        if os.path.exists(json_log_file):
            try:
                with open(json_log_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading existing results: {e}")
        return False

    def evaluate_data(self, data):
        """Evaluates movie data, logs results to a JSON file, and prints summary."""
        json_log_file = f"./results/{self.evaluation_name}.json"
        existing_results = self._load_existing_results(json_log_file)
        if existing_results:
            self.system_prompt = existing_results["system_prompt"]
            eval_duration = existing_results["evaluation_duration"]
            existing_results = existing_results["evaluations"]

        model = genai.GenerativeModel(
            system_instruction=self.system_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature":self.temperature
                }
        )

        valid_outputs = 0
        invalid_outputs = 0
        correct_outputs = 0
        incorrect_outputs = 0
        requests_made = 0
        start_time = time.time()
        last_request_time = start_time

        updated_results = []

        for idx, item in enumerate(data):
            print((idx+1)/len(data))
            participation = item["participation"]
            block = item["block"]

            if existing_results and existing_results[idx]["diversity_score"] != "X":
                updated_results.append(existing_results[idx])
                valid_outputs += 1
                if existing_results[idx]["diversity_score"] == existing_results[idx]["gold"]:
                    correct_outputs += 1
                continue

            current_time = time.time()
            if requests_made >= self.MAX_REQUESTS_PER_MINUTE:
                elapsed_time = current_time - last_request_time
                if elapsed_time < 60:
                    time_to_wait = 60 - elapsed_time + 10
                    print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                    time.sleep(time_to_wait)
                requests_made = 0
                last_request_time = time.time()

            gold_diversity_score = item["diversity_score"]
            prompt = ""

            if existing_results and existing_results[idx]["diversity_score"] == "X":
                prompt = existing_results[idx].get("prompt")
            else:
                prompt_parts = []
                list_names = ["A", "B", "C", "D", "E", "F"]
                idx = 0
                for _, iteration in item["iterations"].items():
                    list_data = iteration['items']
                    movies_info = self._generate_list_info(list_data)
                    prompt_parts.append(f"List {list_names[idx]}:\n{movies_info}")
                    # if self.include_summary and 'summary' in item[list_name]:
                    #     summary_text = self._generate_summary_text(item[list_name]['summary'])
                    #     prompt_parts.append(f"\nSummary:\n{summary_text}")
                    idx += 1

                prompt = "\n\n".join(prompt_parts)

            try:
                response_step1 = model.generate_content(prompt)
                output = json.loads(response_step1.text.strip())
                predicted_score = "X"

                if all(key in output for key in [
                    "list_A_description", 
                    "list_B_description", 
                    "list_C_description", 
                    "list_D_description", 
                    "list_E_description", 
                    "list_F_description", 
                    "diversity_summarization", 
                    "diversity_score"
                    ]):
                    valid_outputs += 1
                    predicted_score = output["diversity_score"]
                    correctness = predicted_score == gold_diversity_score
                    if correctness:
                        print("-OK-")
                        correct_outputs += 1
                    else:
                        incorrect_outputs += 1
                else:
                    invalid_outputs += 1
                    output = {
                        "comparison": "X",
                        "diversity_score": "X",
                        "error": "Invalid JSON response from model"
                    }

                updated_results.append({
                    "participation": participation,
                    "block": block,
                    "prompt": prompt,
                    "response": output,
                    "gold": gold_diversity_score,
                    "diversity_score": predicted_score,
                    "correct": correctness if "diversity_score" in output else False,
                    "error": output.get("error", None)
                })

            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                updated_results.append({
                    "participation": participation,
                    "block": block,
                    "prompt": prompt,
                    "gold": gold_diversity_score,
                    "diversity_score": "X",
                    "correct": False,
                    "error": str(e)
                })

            requests_made += 1
            time.sleep(self.REQUEST_INTERVAL)

        elapsed_time = time.time() - start_time
        if existing_results:
            elapsed_time += eval_duration
        accuracy_percentage = (correct_outputs / valid_outputs) * 100 if valid_outputs > 0 else 0
            
        result = {
            "name": self.evaluation_name,
            "evaluation_duration": elapsed_time,
            "accuracy": accuracy_percentage,
            "system_prompt": model._system_instruction.parts[0].text,
            "evaluations": updated_results
        }

        folder_path = os.path.dirname(json_log_file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(json_log_file, "w") as json_log:
            json.dump(result, json_log, indent=4)

        print("Finished")
        print(f"Accuracy: {accuracy_percentage:.2f}%")
