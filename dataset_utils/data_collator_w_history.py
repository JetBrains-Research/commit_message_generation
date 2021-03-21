from dataclasses import dataclass
from typing import List, Union, Dict, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorWithHistory:
    """
    Data collator used to collate author's history with current message.
    - diff_input_ids, diff_attention_mask: simply concatenate into tensor
    - msg_input_ids, msg_attention_mask: history + current message
    - msg_labels: -100 on history part to avoid computing loss
    - generation_input_ids, generation_attention_mask: history
    """

    tokenizer: PreTrainedTokenizerBase
    max_len: Optional[int] = None

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]],
                                                 List[List[List[int]]],
                                                 torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists
        history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists of lists D:

        all_ids = []                   # input for training or metrics: history + cur_msg
        all_generation_ids = []        # 'prompt' for generation: history
        all_labels = []                # -100 on history & padding to avoid computing loss
        all_generation_labels = []
        all_masks = []                 # 0 on pad tokens and 1 otherwise (right-side padding)
        all_generation_masks = []      # 0 on pad tokens and 1 otherwise (left-side padding)

        # concatenate history examples with current input ids (checking that resulting length is <= max_len)
        for message_ids, history_ids in zip(message_inputs, history_inputs):
            cur_ids = [message_ids]
            cur_labels = [message_ids]
            cur_generation_labels = [message_ids]
            cur_generation_ids = []
            cur_len = len(message_ids)

            for history_input_ids in history_ids[::-1]:
                if cur_len + len(history_input_ids) > self.max_len:
                    break
                cur_len += len(history_input_ids)

                if len(history_input_ids) > 0:
                    cur_ids.insert(0, history_input_ids + self.tokenizer(r' \n ').input_ids)
                    cur_generation_ids.insert(0, history_input_ids + self.tokenizer(r' \n ').input_ids)
                cur_labels.insert(0, [-100 for _ in history_input_ids])
                cur_generation_labels.insert(0, [-100 for _ in history_input_ids])

            # flatten everything into one sequence and convert to tensor of torch.int64
            cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_generation_ids = torch.tensor([ex for sublist in cur_generation_ids for ex in sublist],
                                              dtype=torch.int64)
            if len(cur_generation_ids.size()) == 0:
                cur_generation_ids = torch.tensor([self.tokenizer.bos_token_id])

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

        gen_max_len = max(len(tensor) for tensor in all_generation_ids)
        input_max_len = max(len(tensor) for tensor in all_ids)

        # pad tensors to max length in batch
        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        for i, (id_tensor, mask_tensor, labels_tensor, gen_ids_tensor,
                gen_mask_tensor, gen_labels_tensor) in \
                enumerate(zip(all_ids, all_masks, all_labels,
                              all_generation_ids, all_generation_masks, all_generation_labels)):
            # pad ids with pad_token_id (which doesn't really matter for GPT-2)
            all_ids[i] = torch.nn.functional.pad(id_tensor, pad=[0, input_max_len - id_tensor.numel()], mode='constant',
                                                 value=self.tokenizer.pad_token_id)

            all_generation_ids[i] = torch.nn.functional.pad(gen_ids_tensor,
                                                            pad=[gen_max_len - gen_ids_tensor.numel(), 0],
                                                            mode='constant', value=self.tokenizer.pad_token_id)

            # pad labels with -100
            all_labels[i] = torch.nn.functional.pad(labels_tensor, pad=[0, input_max_len - labels_tensor.numel()],
                                                    mode='constant', value=-100)

            all_generation_labels[i] = torch.nn.functional.pad(gen_labels_tensor,
                                                               pad=[input_max_len - labels_tensor.numel(), 0],
                                                               mode='constant', value=-100)

            # pad masks with zeros
            all_masks[i] = torch.nn.functional.pad(mask_tensor, pad=[0, input_max_len - mask_tensor.numel()],
                                                   mode='constant', value=0)
            all_generation_masks[i] = torch.nn.functional.pad(gen_mask_tensor,
                                                              pad=[gen_max_len - gen_mask_tensor.numel(), 0],
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
    from transformers import GPT2Tokenizer, RobertaTokenizer
    from torch.utils.data import DataLoader

    diff_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    msg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    msg_tokenizer.pad_token = msg_tokenizer.unk_token

    test_dataset = CMGDatasetWithHistory.load_data(diff_tokenizer, msg_tokenizer,
                                                   path='../raw_data/CleanedJiang/test.csv')

    data_collator = DataCollatorWithHistory(tokenizer=msg_tokenizer, max_len=1024)

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

    for batch in test_dataloader:
        print("Message (history + cur_msg)")
        print(msg_tokenizer.batch_decode(batch['msg_input_ids'], skip_special_tokens=True))
        print("Generation (history)")
        print(msg_tokenizer.batch_decode(batch['generation_input_ids'], skip_special_tokens=False))
        print()