from typing import Dict, List

from .base_preprocessor import BasePreprocessor
from .reused_implementations import compute_code_diffs


class RACEPreprocessor(BasePreprocessor):
    def _preprocess_diff(self, header: List[str], diff: str, line_sep: str) -> str:
        """Helper method: transforms single file diff to representation from RACE paper."""
        old_lines = [header[0].strip()]
        new_lines = [header[1].strip() if len(header) == 2 else header[0]]

        for line in diff.split(line_sep):
            line = line.strip()
            if not line:
                continue
            if line.startswith("+"):
                new_lines.extend(line.split(" "))
            elif line.startswith("-"):
                old_lines.extend(line.split(" "))
            else:
                new_lines.extend(line.split(" "))
                old_lines.extend(line.split(" "))
        resulting_tokens: List[str] = compute_code_diffs(old_tokens=old_lines, new_tokens=new_lines)
        return " ".join(resulting_tokens)

    def _preprocess_mods(self, mods: List[Dict[str, str]], line_sep: str = "[NL]", *args, **kwargs) -> str:
        """Transforms a list of file modification made in a commit to a single diff representation from RACE paper."""
        diff = []

        for i, mod in enumerate(mods):
            if mod["change_type"] == "UNKNOWN":
                continue
            elif mod["change_type"] == "ADD":
                header = [f"new file {mod['new_path']}"]
            elif mod["change_type"] == "DELETE":
                header = [f"deleted file {mod['old_path']}"]
            elif mod["change_type"] == "RENAME":
                header = [f"rename from {mod['old_path']}", f"rename to {mod['new_path']}"]
            elif mod["change_type"] == "COPY":
                header = [f"copy from {mod['old_path']}", f"copy to {mod['new_path']}"]
            else:
                header = [f"{mod['new_path']}"]
            diff.append(self._preprocess_diff(header, mod["diff"], line_sep=line_sep))
        return line_sep.join(diff)

    def _preprocess_message(self, message: str, **kwargs) -> str:
        """Returns given message without any changes."""
        return message
