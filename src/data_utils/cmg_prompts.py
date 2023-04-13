from typing import Dict, List, Optional


class CMGPrompts:
    """A set of simple prompts for commit message generation for Completion endpoint."""

    @staticmethod
    def zero_shot_simple(diff: str, _: Optional[str] = None) -> str:
        return f"Write a commit message for a given diff.\nDiff:\n{diff}\nCommit message:\n"

    @staticmethod
    def zero_shot_history(diff: str, previous_message: Optional[str] = None) -> str:
        if previous_message:
            return f"Write a commit message for a given diff. Pay attention to structure of previous message.\nPrevious message:{previous_message}\nDiff:\n{diff}\nCommit message:\n"
        return CMGPrompts.zero_shot_simple(diff)

    @staticmethod
    def one_shot(diff: str, diff_example: str, msg_example: str) -> str:
        return f"Diff:\n{diff_example}\nCommit message:\n{msg_example}\nDiff:\n{diff}\nCommit message:\n"


class CMGChatPrompts:
    """A set of simple prompts for commit message generation for ChatCompletion endpoint.

    Based on OpenAI cookbook:
    https://github.com/openai/openai-cookbook/blob/7622aa1d207d1cc8c88b1c4e08b9f78133bcdb25/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """

    @staticmethod
    def zero_shot_simple(diff: str, _: Optional[str] = None) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates commit messages for given diffs.",
            },
            {"role": "user", "content": CMGPrompts.zero_shot_simple(diff)},
        ]

    @staticmethod
    def zero_shot_history(diff: str, previous_message: Optional[str] = None) -> List[Dict[str, str]]:
        if previous_message:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful instruction-following programming assistant that creates commit messages for given diffs.",
                },
                {"role": "user", "content": CMGPrompts.zero_shot_history(diff=diff, previous_message=previous_message)},
            ]
        return CMGChatPrompts.zero_shot_simple(diff)

    @staticmethod
    def one_shot(diff: str, diff_example: str, msg_example: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful programming assistant that creates commit messages for given diffs.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": CMGPrompts.zero_shot_simple(diff_example),
            },
            {"role": "system", "name": "example_assistant", "content": msg_example},
            {"role": "user", "content": CMGPrompts.zero_shot_simple(diff)},
        ]
