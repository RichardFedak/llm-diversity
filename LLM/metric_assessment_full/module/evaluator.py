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
    def __init__(self, api_key, evaluation_name, system_prompt=None, input_fields=None, include_summary=False):
        """Initializes the MovieEvaluator with API key and configuration."""
        genai.configure(api_key=api_key)
        self.system_prompt = system_prompt
        self.input_fields = input_fields
        self.evaluation_name = evaluation_name
        self.include_summary = include_summary
        self.MAX_REQUESTS_PER_MINUTE = 15
        self.REQUEST_INTERVAL = 60 / self.MAX_REQUESTS_PER_MINUTE
        self.results = []

    def _get_field_value(self, movie, field):
        """Retrieves a field value, handling nested fields."""
        path_parts = field.value.split(".")
        current = movie
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
        return current

    def _generate_list_info(self, list_data):
        list_info = []
        for movie in list_data:
            movie_info = []
            for field in self.input_fields:
                value = self._get_field_value(movie, field)
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

    def evaluate_data(self, data, gold_standard_field='selected_list'):
        """Evaluates movie data, logs results to a JSON file, and prints summary."""
        json_log_file = f"{self.evaluation_name}_results.json"
        existing_results = self._load_existing_results(json_log_file)
        existing_results_map = {}
        if existing_results:
            self.system_prompt = existing_results["system_prompt"]
            existing_results_map = {res["participation"]: res for res in existing_results["evaluations"]}

        model = genai.GenerativeModel(
            system_instruction=self.system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        # Map participation IDs to existing outputs

        valid_outputs = 0
        invalid_outputs = 0
        correct_outputs = 0
        incorrect_outputs = 0
        requests_made = 0
        start_time = time.time()
        last_request_time = start_time

        updated_results = []

        for idx, item in enumerate(data):
            print(idx)
            if idx>1:
                break
            participation = item["participation"]

            if participation in existing_results_map and existing_results_map[participation].get("output") != "X":
                updated_results.append(existing_results_map[participation])
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

            gold_most_diverse = item[gold_standard_field]
            prompt = ""

            if participation in existing_results_map and existing_results_map[participation].get("output") == "X":
                prompt = existing_results_map[participation].get("prompt")
            else:
                prompt_parts = []
                for list_name in ['list_A', 'list_B', 'list_C']:
                    list_data = item[list_name]['items']
                    movies_info = self._generate_list_info(list_data)
                    prompt_parts.append(f"List {list_name[5]}:\n{movies_info}")
                    if self.include_summary and 'summary' in item[list_name]:
                        summary_text = self._generate_summary_text(item[list_name]['summary'])
                        prompt_parts.append(f"\nSummary:\n{summary_text}")

                prompt = "\n\n".join(prompt_parts)

            try:
                response_step1 = model.generate_content(prompt)
                output = json.loads(response_step1.text.strip())

                if all(key in output for key in [
                    "list_A_description", 
                    "list_B_description", 
                    "list_C_description", 
                    "comparison", 
                    "most_diverse_list_reasoning", 
                    "most_diverse_list"
                    ]):
                    valid_outputs += 1
                    correctness = output["most_diverse_list"] == gold_most_diverse
                    if correctness:
                        print("-OK-")
                        correct_outputs += 1
                    else:
                        incorrect_outputs += 1
                else:
                    invalid_outputs += 1
                    output = {
                        "comparison": "X",
                        "most_diverse_list_reasoning": "Invalid Response",
                        "most_diverse_list": "X",
                        "error": "Invalid JSON response from model"
                    }

                updated_results.append({
                    "participation": participation,
                    "prompt": prompt,
                    "response": output,
                    "gold": gold_most_diverse,
                    "output": output.get("most_diverse_list", "X"),
                    "correct": correctness if "most_diverse_list" in output else False,
                    "error": output.get("error", None)
                })

            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                updated_results.append({
                    "participation": participation,
                    "prompt": prompt,
                    "gold": gold_most_diverse,
                    "output": "X",
                    "correct": False,
                    "error": str(e)
                })

            requests_made += 1
            time.sleep(self.REQUEST_INTERVAL)

        elapsed_time = time.time() - start_time
        if existing_results:
            elapsed_time += existing_results["evaluation_duration"]

        result = {
            "name": self.evaluation_name,
            "evaluation_duration": elapsed_time,
            "system_prompt": model._system_instruction.parts[0].text,
            "evaluations": updated_results
        }

        with open(json_log_file, "w") as json_log:
            json.dump(result, json_log, indent=4)

        accuracy_percentage = (correct_outputs / valid_outputs) * 100 if valid_outputs > 0 else 0
        print("Finished")
        print(f"Accuracy: {accuracy_percentage:.2f}%")
