import logging
from itertools import zip_longest
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
            try:
                predictions = self.get_completion([example["prompt"] for example in prompts_chunk])
                with jsonlines.open(output_path, "a") as writer:
                    writer.write_all(
                        [
                            {"Prediction": prediction, "Target": example["target"]}
                            for prediction, example in zip(predictions, prompts_chunk)
                        ]
                    )
            except openai.error.APIError:
                logging.exception("Encountered API error")
                writer.write_all([{"Prediction": None, "Target": example["target"]} for example in prompts_chunk])

    def get_completion_chat_file(self, input_path: str, output_path: str) -> None:
        """Iterates over given file with prompts and saves results from ChatCompletion endpoint for each prompt.

        Args:
            input_path: Path to input file. The input file is expected to be in JSONLines format
              with keys `messages` and `target` for each example.
            output_path: Path to file to write results to. The output file will be in JSONLines format
              with keys `Prediction` and `Target` for each example.
        """
        with jsonlines.open(input_path, "r") as reader:
            for i, line in tqdm(enumerate(reader), "Generating predictions"):
                with jsonlines.open(output_path, "a") as writer:
                    try:
                        writer.write(
                            {"Prediction": self.get_chat_completion(line["messages"]), "Target": line["target"]}
                        )
                    except openai.error.APIError:
                        logging.exception(f"Encountered API error for example {i}")
                        writer.write({"Prediction": None, "Target": line["target"]})

    def fill_completion_chat_file(self, input_path_prompts: str, input_path_predictions: str, output_path: str) -> None:
        """Given input file with prompts and unfinished file with predictions,
        generates a completion for each missed example.

        Use-case: previous run didn't finish completely due to API errors :(

        Args:
            input_path_prompts: Path to input file with prompts. It is expected to be in JSONLines format
              with keys `messages` and `target` for each example.
            input_path_predictions: Path to unfinished file with predictions. It is expected to be in JSONLines format
              with keys `Prediction` and `Target` for each example.
            output_path: Path to file to write results to. The output file will be in JSONLines format
              with keys `Prediction` and `Target` for each example.
        """
        with jsonlines.open(input_path_prompts, "r") as reader_prompts:
            with jsonlines.open(input_path_predictions, "r") as reader_predictions:
                for i, (line_prompts, line_predictions) in tqdm(
                    enumerate(zip_longest(reader_prompts, reader_predictions)), "Generating predictions"
                ):
                    # two cases when we don't have predictions: predictions file is SHORTER than prompts file or current prediction is None
                    if not line_predictions or line_predictions["Prediction"] is None:
                        try:
                            output = {
                                "Prediction": self.get_chat_completion(line_prompts["messages"]),
                                "Target": line_prompts["target"],
                            }
                        except openai.error.APIError:
                            logging.exception(f"Encountered API error for example {i}")
                            output = {"Prediction": None, "Target": line_prompts["target"]}
                    else:
                        output = line_predictions

                    with jsonlines.open(output_path, "a") as writer:
                        writer.write(output)
