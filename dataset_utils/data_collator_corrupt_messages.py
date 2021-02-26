import random
from dataclasses import dataclass
from typing import List, Union, Dict, Any, Tuple

import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary.
    Copied from transformers.data.data_collator"""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


def tolist(x: Union[List[Any], torch.Tensor]):
    """Copied from transformers.data.data_collator"""
    return x.tolist() if isinstance(x, torch.Tensor) else x


@dataclass
class DataCollatorCorruptMessages(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    in my case, the input is diff concatenated with message and I need to mask only the message part
    """

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Copied from transformers.data.data_collator.DataCollatorForWholeWordMask with slight changes
        to mask only the message part.
        """

        message_inputs = [e["message_input_ids"] for e in examples]
        diff_inputs = [e["diff_input_ids"] for e in examples]

        message_batch_input = _collate_batch(message_inputs, self.tokenizer)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["message_input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)
            mask_labels.append(self._whole_word_mask(ref_tokens))

        batch_mask = _collate_batch(mask_labels, self.tokenizer)
        message_inputs, message_labels = self.mask_tokens(message_batch_input, batch_mask)

        input_ids = []
        labels = []
        for diff_ids, message_ids, mask_label in zip(diff_inputs, message_inputs, mask_labels):
            if len(diff_ids) + len(message_ids) > 511:
                diff_ids = diff_ids[:511 - len(message_ids)]

            input_ids.append(torch.cat((torch.tensor(diff_ids),
                                        torch.tensor([self.tokenizer.eos_token_id]), message_ids[1:])))
            labels.append(torch.cat((torch.tensor([-100 for _ in diff_ids]), torch.tensor(mask_label))))

        max_len = max(len(tensor) for tensor in input_ids)

        # pad ids with pad_token_id and labels with -100
        input_ids = [torch.nn.functional.pad(id, pad=(0, max_len - id.numel()), mode='constant',
                                             value=self.tokenizer.pad_token_id) for id in input_ids]
        labels = [torch.nn.functional.pad(label, pad=(0, max_len - label.numel()), mode='constant', value=-100)
                  for label in labels]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.logical_not(input_ids == self.tokenizer.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "target_input_ids": message_batch_input}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        Copied from transformers.data.data_collator.DataCollatorForWholeWordMask
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        Copied from transformers.data.data_collator.DataCollatorForWholeWordMask
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


if __name__ == '__main__':
    import os
    from dataset_utils.cmg_dataset import CMGDataset
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    test_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'test'),
                                        diff_max_len=110, msg_max_len=30, verbose=True)

    data_collator = DataCollatorCorruptMessages(tokenizer=tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

    for batch in test_dataloader:
        print(tokenizer.batch_decode(batch['input_ids']))
        print(tokenizer.batch_decode(batch['target_input_ids']))
        break
