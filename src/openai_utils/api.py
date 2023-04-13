from typing import Dict, List

import backoff
import jsonlines
import openai
from tqdm import tqdm

from conf.openai_config import GenerationConfig


class OpenAIUtils:
    """A simple wrapper to access OpenAI Completion and ChatCompletion endpoints.

    Uses some tips from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb.
    However, currently, it doesn't support async/multiprocessing.
    """

    def __init__(self, model_id: str, generation_kwargs: GenerationConfig):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def get_completion(self, prompts: List[str]) -> str:
        response = openai.Completion.create(
            model=self._model_id, prompt=prompts, **self._generation_kwargs  # type: ignore[arg-type]
        )
        return response.choices[0]["text"]

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def get_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response = openai.ChatCompletion.create(model=self._model_id, messages=messages, **self._generation_kwargs)  # type: ignore[arg-type]
        return response.choices[0]["message"]["content"]

    def get_completion_file(self, input_path: str, output_path: str, chunksize: int = 10) -> None:
        """Iterates over given file with prompts and saves results from Completion endpoint for each prompt.

        Args:
            input_path: Path to input file. The input file is expected to be in JSONLines format
              with keys `prompt` and `target` for each example.
            output_path: Path to file to write results to. The output file will be in JSONLines format
              with keys `Prediction` and `Target` for each example.
            chunksize: Number of examples to include in a single request.
        """
        prompts_chunk: List[Dict[str, str]] = []

        with jsonlines.open(input_path, "r") as reader:
            for line in tqdm(reader, "Generating predictions"):

                if len(prompts_chunk) > chunksize:
                    predictions = self.get_completion([example["prompt"] for example in prompts_chunk])
                    with jsonlines.open(output_path, "a") as writer:
                        writer.write_all(
                            [
                                {"Prediction": prediction, "Target": example["target"]}
                                for prediction, example in zip(predictions, prompts_chunk)
                            ]
                        )
                prompts_chunk.append({"prompt": line["prompt"], "Target": line["target"]})

        if len(prompts_chunk) > 0:
            predictions = self.get_completion([example["prompt"] for example in prompts_chunk])
            with jsonlines.open(output_path, "a") as writer:
                writer.write_all(
                    [
                        {"Prediction": prediction, "Target": example["target"]}
                        for prediction, example in zip(predictions, prompts_chunk)
                    ]
                )

    def get_completion_chat_file(self, input_path: str, output_path: str) -> None:
        """Iterates over given file with prompts and saves results from ChatCompletion endpoint for each prompt.

        Args:
            input_path: Path to input file. The input file is expected to be in JSONLines format
              with keys `messages` and `target` for each example.
            output_path: Path to file to write results to. The output file will be in JSONLines format
              with keys `Prediction` and `Target` for each example.
        """
        with jsonlines.open(input_path, "r") as reader:
            for line in tqdm(reader, "Generating predictions"):
                with jsonlines.open(output_path, "a") as writer:
                    writer.write({"Prediction": self.get_chat_completion(line["messages"]), "Target": line["target"]})
