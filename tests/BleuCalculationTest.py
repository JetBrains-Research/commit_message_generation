import unittest
from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from experiments.BleuCalculation import BleuCalculation
import torch
from transformers import RobertaTokenizer


class BleuChecking(unittest.TestCase):
    CONFIG = {'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl'}
    BLEU_CALCULATION_EXPERIMENT = BleuCalculation(CONFIG)
    tok = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    def test_same_predicted_as_target(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                "Fix typo".split()
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_bleu_zero(self):
        predictions = [
            [
                'Completely incorrect guess .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Totally incorrect guess .'.split()
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_no_predictions(self):
        predictions = [
            [], []
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_same_predicted_as_target_bleu_score(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                "Fix typo".split()
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(100.0, bleu_score)

    def test_bleu_zero_bleu_score(self):
        predictions = [
            [
                'Completely incorrect guess .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Totally incorrect guess .'.split()
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)

    def test_no_predictions_bleu_score(self):
        predictions = [
            [], []
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, \
                                                 trg_encodings=BleuChecking.tok(
                                                     ['Fix android_media_AudioSystem_getMasterMute return type .',
                                                      'Fix typo'], padding=True, return_tensors='pt'))
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)


if __name__ == '__main__':
    unittest.main()
