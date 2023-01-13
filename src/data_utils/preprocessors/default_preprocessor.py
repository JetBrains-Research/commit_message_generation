from typing import Dict, List

from .base_preprocessor import BasePreprocessor


class DefaultPreprocessor(BasePreprocessor):
    def _preprocess_diff(self, diff: str, line_sep: str, **kwargs) -> str:
        """Return given diff without any changes."""
        return diff

    def _preprocess_mods(self, mods: List[Dict[str, str]], line_sep: str = "[NL]", **kwargs) -> str:
        """
        Transforms a list of all files modifications made in a commit into a single string representation.

        Specifically, adds a header to each file diff (https://git-scm.com/docs/git-diff#_generating_patch_text_with_p)
        and concatenates the results.

        Args:
            mods: A list of files modifications made in a commit.
            line_sep: Line separator in diffs.

        Returns:
            A single string representation of all files modifications made in a commit.
        """
        diff = ""

        for mod in mods:
            if mod["change_type"] == "UNKNOWN":
                continue
            elif mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}"
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}"
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}{line_sep}rename to {mod['new_path']}"
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}{line_sep}copy to {mod['new_path']}"
            else:
                file_diff = f"{mod['new_path']}"
            diff += file_diff + line_sep + self._preprocess_diff(mod["diff"], line_sep=line_sep)

        return diff

    def _preprocess_message(self, message: str, **kwargs) -> str:
        """Returns given message without any changes."""
        return message
