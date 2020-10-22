from typing import List

from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

from models.EncoderDecoder import EncoderDecoder
from Config import Config
from models.Decoder.search import create_decode_method
from models.train_utils import rebatch, calculate_top_k_accuracy, \
    create_greedy_decode_method_with_batch_support


class AccuracyCalculation:
    def __init__(self, model: EncoderDecoder, tokenizer: RobertaTokenizer, max_len: int,
                 greedy: bool, config: Config) -> None:
        super().__init__()
        self.model = model
        self.trg_vocab = tokenizer.vocab_size
        self.pad_index: int = tokenizer.pad_token_id
        sos_index: int = tokenizer.eos_token_id
        self.eos_index: int = tokenizer.eos_token_id
        self.config = config
        self.beam_size = self.config['BEAM_SIZE']
        self.topk_values = [1] if greedy else self.config['TOP_K']
        if greedy:
            self.decode_method = create_greedy_decode_method_with_batch_support(
                self.model, max_len, sos_index, self.eos_index,
                self.trg_vocab.unk_index, len(self.trg_vocab)
            )
        else:
            self.decode_method = create_decode_method(
                self.model, max_len, sos_index, self.eos_index,
                self.trg_vocab.unk_index, len(self.trg_vocab), self.beam_size,
                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                verbose=False
            )
        self.batch_size = self.config['TEST_BATCH_SIZE'] if greedy else 1

    def conduct(self, dataset: Dataset, dataset_label: str) -> List[List[List[str]]]:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...')
        data_iterator = DataLoader(dataset, batch_size=self.batch_size)
        correct_all_k, total, max_top_k_predicted = \
            calculate_top_k_accuracy(self.topk_values,
                                     [rebatch(self.pad_index, batch, dataset, self.config) for batch in data_iterator],
                                     self.decode_method, self.trg_vocab, self.eos_index, len(dataset))
        for correct_top_k, k in zip(correct_all_k, self.topk_values):
            print(f'Top-{k} accuracy: {correct_top_k} / {total} = {correct_top_k / total}')
        return max_top_k_predicted
