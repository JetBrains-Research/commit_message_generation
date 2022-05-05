from typing import Dict, List

import datasets

from src.metrics.reused_implementations import bleuFromMaps, splitPuncts

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
B-Norm is a variation of BLEU. It uses smoothing by Lin and Och, 2004 and does some additional preprocessing steps.
It was recommended for evaluation of commit message generation approaches in the 
"On the Evaluation of Commit Message Generation Models: An Experimental Study" paper accepted to ICSME 2021.
This class uses implementation provided in the replication package.
"""


class BLEUNorm(datasets.Metric):
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
            codebase_urls=["https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical/blob/main/metrics/B-Norm.py"],
        )

    def _compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:

        prediction_map = {i: [splitPuncts(pred.strip().lower())] for i, pred in enumerate(predictions)}
        gold_map = {i: [splitPuncts(ref.strip().lower())] for i, ref in enumerate(references)}

        return {"b_norm": bleuFromMaps(gold_map, prediction_map)[0] / 100.0}
