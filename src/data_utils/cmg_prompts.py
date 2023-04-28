from typing import Dict, List, Optional


class CMGPrompts:
    """A set of simple prompts for commit message generation for Completion endpoint."""

    @staticmethod
    def zero_shot_simple(diff: str, **kwargs) -> str:
        return f"Write a commit message for a given diff.\nDiff:\n{diff}\nCommit message:\n"

    @staticmethod
    def zero_shot_simple_completion(diff: str, prefix: str, **kwargs) -> str:
        return f"Complete a given commit message prefix for a given diff.\nDiff:\n{diff}\nCommit message:\n{prefix}"

    @staticmethod
    def zero_shot_history(diff: str, previous_message: Optional[str] = None) -> str:
        if previous_message:
            return f"Write a commit message for a given diff. Pay attention to structure of previous message.\nPrevious message:{previous_message}\nDiff:\n{diff}\nCommit message:\n"
        return CMGPrompts.zero_shot_simple(diff=diff)

    @staticmethod
    def zero_shot_history_completion(diff: str, prefix: str, previous_message: Optional[str] = None) -> str:
        if previous_message:
            return f"Complete a given commit message prefix for a given diff. Pay attention to structure of previous message.\nPrevious message:{previous_message}\nDiff:\n{diff}\nCommit message:\n{prefix}"
        return CMGPrompts.zero_shot_simple_completion(diff=diff, prefix=prefix)


class CMGChatPrompts:
    """A set of simple prompts for commit message generation for ChatCompletion endpoint.

    Based on OpenAI cookbook:
    https://github.com/openai/openai-cookbook/blob/7622aa1d207d1cc8c88b1c4e08b9f78133bcdb25/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """

    @staticmethod
    def zero_shot_simple(diff: str, **kwargs) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates commit messages for given diffs.",
            },
            {"role": "user", "content": CMGPrompts.zero_shot_simple(diff=diff)},
        ]

    @staticmethod
    def zero_shot_simple_completion(diff: str, prefix: str, **kwargs) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that autocompletes commit messages for given diffs.",
            },
            {"role": "user", "content": CMGPrompts.zero_shot_simple_completion(diff=diff, prefix=prefix)},
        ]

    @staticmethod
    def zero_shot_history(diff: str, previous_message: Optional[str] = None) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful instruction-following programming assistant that creates commit messages for given diffs.",
            },
            {
                "role": "user",
                "content": CMGPrompts.zero_shot_history(diff=diff, previous_message=previous_message),
            },
        ]

    @staticmethod
    def zero_shot_history_completion(
        diff: str, prefix: str, previous_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful instruction-following programming assistant that autocompletes commit messages for given diffs.",
            },
            {
                "role": "user",
                "content": CMGPrompts.zero_shot_history_completion(
                    diff=diff, prefix=prefix, previous_message=previous_message
                ),
            },
        ]
