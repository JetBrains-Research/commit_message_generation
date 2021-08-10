import datasets
from typing import List, Dict
from nltk.translate.chrf_score import corpus_chrf


class ChrF(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="token"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="token"), id="sequence"),
                }
            ),
            codebase_urls=["https://www.nltk.org/_modules/nltk/translate/chrf_score.html#corpus_chrf"],
        )

    def _compute(
        self, predictions: List[List[str]], references: List[List[str]], min_len=1, max_len=6, beta=3.0
    ) -> Dict[str, float]:
        chrf = corpus_chrf(references=references, hypotheses=predictions, min_len=min_len, max_len=max_len, beta=beta)
        return {"chrf": chrf}
