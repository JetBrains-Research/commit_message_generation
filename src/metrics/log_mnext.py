import datasets
import numpy as np
from typing import List, Dict
from src.metrics.reused_implementations import log_mnext_score


_CITATION = """\
@inproceedings{tao2021evaluation,
  title={On the Evaluation of Commit Message Generation Models: An Experimental Study},
  author={Tao, Wei and Wang, Yanlin and Shi, Ensheng and Du, Lun and Han, Shi and Zhang, Hongyu and Zhang, Dongmei and Zhang, Wenqiang},
  booktitle={2021 IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  pages={126--136},
  year={2021},
  organization={IEEE}
}
@inproceedings{Papineni02bleu:a,
    author = {Kishore Papineni and Salim Roukos and Todd Ward and Wei-jing Zhu},
    title = {BLEU: a Method for Automatic Evaluation of Machine Translation},
    booktitle = {},
    year = {2002},
    pages = {311--318}
}
@inproceedings{lin-och-2004-orange,
    title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
    author = "Lin, Chin-Yew  and
      Och, Franz Josef",
    booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
    month = "aug 23{--}aug 27",
    year = "2004",
    address = "Geneva, Switzerland",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/C04-1072",
    pages = "501--507",
}
"""

_DESCRIPTION = """\
"""


class LogMNEXT(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[
                "https://github.com/CMGeval/Evaluating-CMG/blob/main/Experimental%20results/The%20Log-MNEXT%20metric.py"
            ],
        )

    def _compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:

        scores = [log_mnext_score([ref], pred) for ref, pred in zip(references, predictions)]

        return {"log_mnext": np.mean(scores)}
