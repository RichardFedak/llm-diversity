from ollama import chat
from typing import Type, Optional, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class OllamaWrapper(Generic[T]):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "llama3.2",
        response_model: Optional[Type[T]] = None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.response_model = response_model

    def ask(self, prompt: str) -> T:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = chat(
            messages=messages,
            model=self.model_name,
            format=self.response_model.model_json_schema() if self.response_model else None,
        )

        if self.response_model:
            return self.response_model.model_validate_json(response.message.content)
        return response.message.content
