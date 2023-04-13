import logging
from typing import Dict, List, Tuple

import jsonlines
import tiktoken
from tqdm import tqdm


class TokenEstimator:
    """A class used to estimate the expected number of tokens for queries for OpenAI ChatCompletion or Completion endpoints.

    Based on OpenAI cookbook:
    https://github.com/openai/openai-cookbook/blob/7622aa1d207d1cc8c88b1c4e08b9f78133bcdb25/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """

    def __init__(self, model_id: str):
        self.tokenizer = TokenEstimator._get_tokenizer_for_model(model_id)
        self._tokens_per_message, self._tokens_per_name = TokenEstimator._get_model_specific_num_tokens(model_id)

    @staticmethod
    def _get_tokenizer_for_model(model_id: str) -> tiktoken.Encoding:
        try:
            encoding = tiktoken.encoding_for_model(model_id)
        except KeyError:
            logging.warning("Model {model_id} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        return encoding

    @staticmethod
    def _get_model_specific_num_tokens(model_id: str) -> Tuple[int, int]:
        if model_id == "gpt-3.5-turbo":
            logging.warning("gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model_id == "gpt-4":
            logging.warning("gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model_id}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        return tokens_per_message, tokens_per_name

    def get_num_tokens_chat(self, messages: List[Dict[str, str]]) -> int:
        """Returns the expected number of tokens for messages for ChatCompletion endpoint."""
        num_tokens = 0
        for message in messages:
            num_tokens += self._tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                if key == "name":
                    num_tokens += self._tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_num_tokens(self, messages: List[str]) -> int:
        """Returns the expected number of tokens for messages for Completion endpoint."""
        return sum(len(encoded_msg) for encoded_msg in self.tokenizer.encode_batch(messages))

    def get_num_tokens_file(self, input_path: str, num_tokens_to_generate: int) -> Tuple[int, int]:
        """Estimate the expected number of tokens for all prompts in the given file.

        Args:
            input_path: Path to input file. The input file is expected to be in JSONLines format
              with keys `prompt`, `messages` and `target` for each example.
            num_tokens_to_generate: Maximum number of tokens that will be generated for each example.

        Returns:
            A tuple of two integers: expected number of tokens for prompts & expected number of tokens for completion.
        """
        total = 0
        num_examples = 0
        with jsonlines.open(input_path, "r") as reader:
            for line in tqdm(reader, "Calculating number of tokens that processing of this file will consume"):
                if line["prompt"] is not None:
                    assert line["messages"] is None
                    total += self.get_num_tokens([line["prompt"]])
                if line["messages"] is not None:
                    assert line["prompt"] is None
                    total += self.get_num_tokens_chat(line["messages"])
                num_examples += 1
        return total, num_examples * num_tokens_to_generate
