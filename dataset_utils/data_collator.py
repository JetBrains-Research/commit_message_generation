from dataclasses import dataclass
from typing import List, Union, Dict, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollator:
    """
    Data collator used to collate repo history with each message
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for generation
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]], List[List[List[int]]]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Copied from transformers.data.data_collator.DataCollatorForWholeWordMask with slight changes
        to mask only the message part.
        """

        message_inputs = [e["msg_input_ids_unprocessed"] for e in examples]

        # use only first part of message for generation
        all_generation_ids = []
        all_generation_masks = []
        all_generation_labels = []

        all_message_ids = []
        all_message_masks = []
        for message_ids in message_inputs:
            if len(message_ids) > 8:
                cur_generation_ids = [message_ids[:len(message_ids) - 5]]
                cur_generation_labels = [[- 100 for _ in message_ids[:len(message_ids) - 5]]]
                cur_generation_labels[0].extend(message_ids[len(message_ids) - 5:])

            else:
                cur_generation_ids = [message_ids[:len(message_ids) // 2]]
                cur_generation_labels = [[-100 for _ in message_ids[:len(message_ids) // 2]]]
                cur_generation_labels[0].extend(message_ids[len(message_ids) // 2:])

            # flatten everything into one sequence and convert to tensor of torch.int
            cur_generation_ids = torch.tensor([ex for sublist in cur_generation_ids for ex in sublist],
                                              dtype=torch.int64)

            cur_generation_labels = torch.tensor([ex for sublist in cur_generation_labels for ex in sublist],
                                                 dtype=torch.int64)
            cur_ids = torch.tensor(message_ids, dtype=torch.int64)

            # create ones for attention mask
            cur_generation_mask = torch.ones_like(cur_generation_ids)
            cur_mask = torch.ones_like(cur_ids)

            all_generation_ids.append(cur_generation_ids)
            all_generation_masks.append(cur_generation_mask)
            all_generation_labels.append(cur_generation_labels)
            all_message_ids.append(cur_ids)
            all_message_masks.append(cur_mask)

        generation_ids_max_len = max(len(tensor) for tensor in all_generation_ids)
        generation_labels_max_len = max(len(tensor) for tensor in all_generation_labels)
        inputs_max_len = max(len(tensor) for tensor in all_message_ids)

        # pad tensors to max length in batch
        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        for i, (generation_ids_tensor, generation_mask_tensor, generation_labels_tensor,
                message_ids_tensor, message_mask_tensor) in \
                enumerate(zip(all_generation_ids, all_generation_masks, all_generation_labels,
                              all_message_ids, all_message_masks)):
            # pad ids with unk_token_id (which doesn't really matter as GPT-2 uses attention mask)
            all_generation_ids[i] = torch.nn.functional.pad(generation_ids_tensor,
                                                            pad=(
                                                                generation_ids_max_len - generation_ids_tensor.numel(),
                                                                0),
                                                            mode='constant', value=self.tokenizer.unk_token_id)

            all_message_ids[i] = torch.nn.functional.pad(message_ids_tensor,
                                                         pad=(0, inputs_max_len - message_ids_tensor.numel()),
                                                         mode='constant', value=self.tokenizer.unk_token_id)

            # pad labels with -100
            all_generation_labels[i] = torch.nn.functional.pad(generation_labels_tensor,
                                                               pad=(
                                                                   generation_labels_max_len - generation_labels_tensor.numel(),
                                                                   0),
                                                               mode='constant', value=-100)

            # pad masks with zeros
            all_generation_masks[i] = torch.nn.functional.pad(generation_mask_tensor,
                                                              pad=(0,
                                                                   generation_ids_max_len - generation_mask_tensor.numel()
                                                                   ),
                                                              mode='constant', value=0)
            all_message_masks[i] = torch.nn.functional.pad(message_mask_tensor,
                                                           pad=(0, inputs_max_len - message_mask_tensor.numel()),
                                                           mode='constant', value=0)

        all_generation_ids = torch.stack(all_generation_ids)
        all_generation_masks = torch.stack(all_generation_masks)
        all_generation_labels = torch.stack(all_generation_labels)
        all_message_ids = torch.stack(all_message_ids)
        all_message_masks = torch.stack(all_message_masks)

        return {"diff_input_ids": torch.stack([e["diff_input_ids"] for e in examples]),
                "diff_attention_mask": torch.stack([e["diff_attention_mask"] for e in examples]),
                "msg_input_ids": all_message_ids,
                "msg_attention_mask": all_message_masks,
                "generation_input_ids": all_generation_ids,
                "generation_attention_mask": all_generation_masks,
                "generation_labels": all_generation_labels}


if __name__ == '__main__':
    import os
    from dataset_utils.cmg_dataset import CMGDataset
    from transformers import RobertaTokenizer, GPT2Tokenizer
    from torch.utils.data import DataLoader

    src_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    trg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    trg_tokenizer.pad_token = trg_tokenizer.unk_token

    test_dataset = CMGDataset.load_data(src_tokenizer, trg_tokenizer,
                                        path=os.path.join('../raw_data/github_data', 'test'),
                                        diff_max_len=110, msg_max_len=30, verbose=True)

    data_collator = DataCollator(tokenizer=trg_tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

    for batch in test_dataloader:
        print(src_tokenizer.batch_decode(batch['diff_input_ids'], skip_special_tokens=True))
        print(trg_tokenizer.batch_decode(batch['generation_input_ids'], skip_special_tokens=True))
        print(batch['msg_input_ids'])
        print(batch['msg_attention_mask'])
        break
