from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy.typing as npt


@dataclass
class RetrievalExample:
    diff_input_ids: List[int] | npt.NDArray
    diff: str
    message: str
    idx: int

    def dict(self):
        return {k: v for k, v in asdict(self).items() if k != "diff_input_ids"}


@dataclass
class RetrievalPrediction:
    message: str
    diff: str
    distance: float

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def mods_to_diff(mods: List[Dict[str, str]], line_sep: str = "[NL]") -> str:
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
        diff += file_diff + line_sep + mod["diff"] + line_sep
    return diff
