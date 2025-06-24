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
            f"Popularity Diversity: {summary['popularity_diversity']['value']}\n"
            f"Genre Diversity: {summary['genre_diversity']['value']}\n"
            f"Theme Diversity: {summary['theme_diversity']['value']}\n"
            f"Time Span: {summary['time_span']['value']}\n"
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

    def _convert_to_indices(self, char_list):
        return [ord(char.upper()) - ord('A') for char in char_list]

    def _get_int(self, value):
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        else:
            return None


    def evaluate_data(self, data, gold_standard_field='selected'):
        """Evaluates movie data, logs results to a JSON file, and prints summary."""
        json_log_file = f"./results/{self.evaluation_name}.json"
        existing_results = self._load_existing_results(json_log_file)
        if existing_results:
            self.system_prompt = existing_results["system_prompt"]
            eval_duration = existing_results["evaluation_duration"]
            existing_results = existing_results["evaluations"]

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-001",
            system_instruction=self.system_prompt,
            generation_config={
                "response_mime_type": "application/json"
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
            print(idx)
            participation = item["participation"]

            if existing_results and existing_results[idx]["final_ordering"] != "X":
                updated_results.append(existing_results[idx])
                valid_outputs += 1
                if existing_results[idx]["final_ordering"] == existing_results[idx]["gold"]:
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

            gold_most_diverse = [int(x.strip()) for x in item[gold_standard_field][1:-1].split(',')]
            prompt = ""

            if existing_results and existing_results[idx]["final_ordering"] == "X":
                prompt = existing_results[idx].get("prompt")
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
                final_ordering = "X"

                if all(key in output for key in [
                    "list_A_description", 
                    "list_B_description", 
                    "list_C_description", 
                    "comparison", 
                    "diversity_scores"
                    ]):
                    valid_outputs += 1
                    final_ordering = self._convert_to_indices(sorted(output["diversity_scores"], key=output["diversity_scores"].get))
                    fixed_order = ["A", "B", "C"]
                    approx_scores = [round((self._get_int(output["diversity_scores"][key]) / 10), 2) for key in fixed_order]
                    correctness = final_ordering == gold_most_diverse
                    if correctness:
                        print("-OK-")
                        correct_outputs += 1
                    else:
                        incorrect_outputs += 1
                else:
                    invalid_outputs += 1
                    output = {
                        "comparison": "X",
                        "diversity_scores": "X",
                        "error": "Invalid JSON response from model"
                    }

                updated_results.append({
                    "participation": participation,
                    "prompt": prompt,
                    "response": output,
                    "gold": gold_most_diverse,
                    "final_ordering": final_ordering,
                    "approx_scores": approx_scores,
                    "correct": correctness if "diversity_scores" in output else False,
                    "error": output.get("error", None)
                })

            except Exception as e:
                print(f"Error generating response for row {idx}: {e}")
                invalid_outputs += 1
                updated_results.append({
                    "participation": participation,
                    "prompt": prompt,
                    "gold": gold_most_diverse,
                    "final_ordering": "X",
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
