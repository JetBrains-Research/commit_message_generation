from typing import Dict, List

from src.metrics.reused_implementations import bleuFromMaps, splitPuncts


def b_norm_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    prediction_map = {i: [splitPuncts(pred.strip().lower())] for i, pred in enumerate(predictions)}
    gold_map = {i: [splitPuncts(ref.strip().lower())] for i, ref in enumerate(references)}
    return bleuFromMaps(gold_map, prediction_map)[0] / 100.0
