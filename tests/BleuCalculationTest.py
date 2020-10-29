import unittest
from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from experiments.BleuCalculation import BleuCalculation
import torch

class BleuChecking(unittest.TestCase):
    CONFIG = {'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl'}
    BLEU_CALCULATION_EXPERIMENT = BleuCalculation(CONFIG)

    def test_same_predicted_as_target(self):
        predictions = [
            [
                [1, 2, 3, 4],
                [0, 1, 2, 3, 4]
            ],
            [
                [1, 5, 600, 600]
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
                       [1, 2, 3, 4],
                       [1, 5, 600, 600]])})
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
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 5, 600, 600]])})
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_only_one_example(self):
        predictions = [
                [1, 2, 3, 4],
                [0, 1, 2, 3, 4]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4]])})
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_no_predictions(self):
        predictions = [
            [], []
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 5, 600, 600]])})
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_same_predicted_as_target_bleu_score(self):
        predictions = [
            [
                [1, 2, 3, 4],
                [0, 1, 2, 3, 4]
            ],
            [
                [1, 5, 600, 600]
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 5, 600, 600]])})
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(100.0, bleu_score)

    def test_bleu_zero_bleu_score(self):
        predictions = [
            [
                [100, 200, 300, 500],
                [0, 1, 2, 3, 4]
            ],
            [
                [5133, 5133]
            ]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 5, 600, 600]])})
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)

    def test_only_one_example_bleu_score(self):
        predictions = [
            [1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4]])})
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(100.0, bleu_score)

    def test_no_predictions_bleu_score(self):
        predictions = [
            [], []
        ]
        dataset = CommitMessageGenerationDataset(src_encodings={}, trg_encodings={'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 5, 600, 600]])})
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)


if __name__ == '__main__':
    unittest.main()
