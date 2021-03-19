from dataclasses import dataclass
from typing import List, Union, Dict, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorWithHistory:
    """
    Data collator used to collate repo history with each message
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for generation
    """

    tokenizer: PreTrainedTokenizerBase
    max_len: Optional[int] = None

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]], List[List[List[int]]]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Copied from transformers.data.data_collator.DataCollatorForWholeWordMask with slight changes
        to mask only the message part.
        """

        message_inputs = [e["message_input_ids"] for e in examples]  # 2D - list of lists
        history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists of lists

        # concatenate history examples with current input ids (checking that resulting length is <= max_len)
        all_ids = []
        all_generation_ids = []
        all_labels = []
        all_generation_labels = []
        all_masks = []
        all_generation_masks = []
        for message_ids, history_ids in zip(message_inputs, history_inputs):
            cur_ids = [message_ids]

            if len(message_ids) > 8:
                cur_generation_ids = [message_ids[:len(message_ids) - 5]]
                cur_generation_labels = [[- 100 for _ in message_ids[:len(message_ids) - 5]]]
                cur_generation_labels[0].extend(message_ids[len(message_ids) - 5:])
            else:
                cur_generation_ids = [message_ids[:len(message_ids) // 2]]
                cur_generation_labels = [[-100 for _ in message_ids[:len(message_ids) // 2]]]
                cur_generation_labels[0].extend(message_ids[len(message_ids) // 2:])

            cur_labels = [message_ids]
            cur_len = len(message_ids)

            for history_input_ids in history_ids[::-1]:
                if cur_len + len(history_input_ids) + 5 > self.max_len:
                    break
                cur_len += len(history_input_ids)
                cur_ids.insert(0, history_input_ids)
                cur_generation_ids.insert(0, history_input_ids)
                cur_labels.insert(0, [-100 for _ in history_input_ids])
                cur_generation_labels.insert(0, [-100 for _ in history_input_ids])

            # flatten everything into one sequence and convert to tensor of torch.int
            cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_generation_ids = torch.tensor([ex for sublist in cur_generation_ids for ex in sublist],
                                              dtype=torch.int64)
            cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)
            cur_generation_labels = torch.tensor([ex for sublist in cur_generation_labels for ex in sublist],
                                                 dtype=torch.int64)
            # create ones for attention mask
            cur_mask = torch.ones_like(cur_ids)
            cur_generation_mask = torch.ones_like(cur_generation_ids)

            all_ids.append(cur_ids)
            all_generation_ids.append(cur_generation_ids)
            all_labels.append(cur_labels)
            all_generation_labels.append(cur_generation_labels)
            all_masks.append(cur_mask)
            all_generation_masks.append(cur_generation_mask)

        generation_max_len = max(len(tensor) for tensor in all_generation_ids)
        input_max_len = max(len(tensor) for tensor in all_ids)

        # pad tensors to max length in batch
        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        for i, (id_tensor, mask_tensor, labels_tensor, generation_ids_tensor,
                generation_mask_tensor, generation_labels_tensor) in \
                enumerate(zip(all_ids, all_masks, all_labels,
                              all_generation_ids, all_generation_masks, all_generation_labels)):
            # pad ids with unk_token_id (which doesn't really matter as GPT-2 uses attention mask)
            all_ids[i] = torch.nn.functional.pad(id_tensor, pad=(0, input_max_len - id_tensor.numel()), mode='constant',
                                                 value=self.tokenizer.unk_token_id)

            all_generation_ids[i] = torch.nn.functional.pad(generation_ids_tensor,
                                                            pad=(generation_max_len - generation_ids_tensor.numel(), 0),
                                                            mode='constant', value=self.tokenizer.unk_token_id)

            # pad labels with -100
            all_labels[i] = torch.nn.functional.pad(labels_tensor, pad=(0, input_max_len - labels_tensor.numel()),
                                                    mode='constant', value=-100)

            all_generation_labels[i] = torch.nn.functional.pad(generation_labels_tensor,
                                                               pad=(
                                                                   input_max_len - generation_labels_tensor.numel(),
                                                                   0),
                                                               mode='constant', value=-100)

            # pad masks with zeros
            all_masks[i] = torch.nn.functional.pad(mask_tensor, pad=(0, input_max_len - mask_tensor.numel()),
                                                   mode='constant', value=0)
            all_generation_masks[i] = torch.nn.functional.pad(generation_mask_tensor,
                                                              pad=(
                                                                  generation_max_len - generation_mask_tensor.numel(),
                                                                  0),
                                                              mode='constant', value=0)

        all_ids = torch.stack(all_ids)
        all_masks = torch.stack(all_masks)
        all_labels = torch.stack(all_labels)
        all_generation_ids = torch.stack(all_generation_ids)
        all_generation_masks = torch.stack(all_generation_masks)
        all_generation_labels = torch.stack(all_generation_labels)

        return {"diff_input_ids": torch.stack([e["diff_input_ids"] for e in examples]),
                "diff_attention_mask": torch.stack([e["diff_attention_mask"] for e in examples]),
                "msg_input_ids": all_ids,
                "msg_attention_mask": all_masks,
                "msg_labels": all_labels,
                "generation_input_ids": all_generation_ids,
                "generation_attention_mask": all_generation_masks,
                "generation_labels": all_generation_labels}


if __name__ == '__main__':
    from dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
    from transformers import GPT2Tokenizer
    from torch.utils.data import DataLoader

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    test_dataset = CMGDatasetWithHistory.load_data(tokenizer, path='../raw_data/github_data',
                                                   diff_max_len=110, msg_max_len=30, verbose=True)

    data_collator = DataCollatorWithHistory(tokenizer=tokenizer, max_len=1024)

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

    for batch in test_dataloader:
        print(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
        print(tokenizer.batch_decode(batch['generation_input_ids'], skip_special_tokens=True))
        break
