from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollator:
    """
    This class is used to construct batch out of list of examples.

    - Commit diffs are simply padded to maximum length.

      Format: `[BOS] diff [EOS]`
    - For training and Next Token Prediction metrics, there is an option to concatenate each commit message
      with corresponding history.

      Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] message [SEP]`

      Format without history: `[BOS] message [SEP]`
    - For generation, there is an option to use corresponding history.

      Format with history: `[BOS] history_1 [SEP] ... history_k [SEP]`

      Format without history: `[BOS]`
    """

    diff_tokenizer: PreTrainedTokenizerBase
    msg_tokenizer: PreTrainedTokenizerBase
    max_len: int
    sep_tokens: List[int]
    with_history: bool
    generation: bool = False
    testing: bool = False

    def __call__(self, examples: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        if not self.testing:
            diff_inputs: List[List[int]] = [e["diff_input_ids"] for e in examples]
            message_inputs: List[List[int]] = [e["msg_input_ids"] for e in examples]
            history_inputs: List[List[List[int]]] = [e["history_input_ids"] for e in examples]

            all_msg_ids = []  # input for training or NTP metrics: history + cur_msg (right-side padding)
            all_msg_masks = []  # 0 on pad tokens and 1 otherwise (right-side padding)
            all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)

            all_generation_ids = []  # 'prompt' for generation: history (left-side padding)
            all_generation_masks = []  # 0 on pad tokens and 1 otherwise (left-side padding)

            # concatenate history examples with current input ids (checking that resulting length is <= max_len)
            for message_ids, history_ids in zip(message_inputs, history_inputs):
                cur_ids = [[self.msg_tokenizer.bos_token_id]]
                cur_labels = [[self.msg_tokenizer.bos_token_id]]
                cur_len = min(self.max_len, len(message_ids) + 1 + len(self.sep_tokens))

                if self.with_history:
                    # insert previous messages from history until we reach max_len
                    for history_input_ids in history_ids[::-1]:
                        if cur_len + len(history_input_ids) + len(self.sep_tokens) > self.max_len:
                            break

                        cur_len += len(history_input_ids) + len(self.sep_tokens)
                        cur_ids.append(history_input_ids + self.sep_tokens)
                        cur_labels.append([-100 for _ in history_input_ids + self.sep_tokens])

                cur_generation_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)

                cur_ids.append(message_ids[: self.max_len - 1 - len(self.sep_tokens)] + self.sep_tokens)
                cur_labels.append(message_ids[: self.max_len - 1 - len(self.sep_tokens)] + self.sep_tokens)

                cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
                cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)

                # create ones for attention mask
                cur_mask = torch.ones_like(cur_ids)
                cur_generation_mask = torch.ones_like(cur_generation_ids)

                all_msg_ids.append(cur_ids)
                all_generation_ids.append(cur_generation_ids)

                all_msg_masks.append(cur_mask)
                all_generation_masks.append(cur_generation_mask)

                all_msg_labels.append(cur_labels)

            all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
            all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

            gen_max_len = max(len(tensor) for tensor in all_generation_ids)
            msg_max_len = max(len(tensor) for tensor in all_msg_ids)
            diff_max_len = max(len(tensor) for tensor in all_diff_ids)

            # pad tensors to max length in batch
            # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
            for i in range(len(all_diff_ids)):
                # pad ids with pad_token_id (which doesn't really matter for GPT-2)
                all_msg_ids[i] = torch.nn.functional.pad(
                    all_msg_ids[i],
                    pad=[0, msg_max_len - all_msg_ids[i].numel()],
                    mode="constant",
                    value=self.msg_tokenizer.pad_token_id,
                )
                all_diff_ids[i] = torch.nn.functional.pad(
                    all_diff_ids[i],
                    pad=[0, diff_max_len - all_diff_ids[i].numel()],
                    mode="constant",
                    value=self.diff_tokenizer.pad_token_id,
                )
                all_generation_ids[i] = torch.nn.functional.pad(
                    all_generation_ids[i],
                    pad=[gen_max_len - all_generation_ids[i].numel(), 0],
                    mode="constant",
                    value=self.msg_tokenizer.pad_token_id,
                )

                # pad labels with -100
                all_msg_labels[i] = torch.nn.functional.pad(
                    all_msg_labels[i], pad=[0, msg_max_len - all_msg_labels[i].numel()], mode="constant", value=-100
                )

                # pad masks with zeros
                all_msg_masks[i] = torch.nn.functional.pad(
                    all_msg_masks[i], pad=[0, msg_max_len - all_msg_masks[i].numel()], mode="constant", value=0
                )
                all_diff_masks[i] = torch.nn.functional.pad(
                    all_diff_masks[i], pad=[0, diff_max_len - all_diff_masks[i].numel()], mode="constant", value=0
                )
                all_generation_masks[i] = torch.nn.functional.pad(
                    all_generation_masks[i],
                    pad=[gen_max_len - all_generation_masks[i].numel(), 0],
                    mode="constant",
                    value=0,
                )

            all_diff_ids = torch.stack(all_diff_ids)
            all_diff_masks = torch.stack(all_diff_masks)

            all_msg_ids = torch.stack(all_msg_ids)
            all_msg_masks = torch.stack(all_msg_masks)
            all_msg_labels = torch.stack(all_msg_labels)

            all_generation_ids = torch.stack(all_generation_ids)
            all_generation_masks = torch.stack(all_generation_masks)

            batch_result = {
                "diff_input_ids": all_diff_ids,
                "diff_attention_mask": all_diff_masks,
                "msg_input_ids": all_msg_ids,
                "msg_attention_mask": all_msg_masks,
                "msg_labels": all_msg_labels,
            }
            if self.generation:
                batch_result.update(
                    {
                        "generation_input_ids": all_generation_ids,
                        "generation_attention_mask": all_generation_masks,
                    }
                )

            return batch_result
        else:
            batch_size = len(examples)
            return {
                "diff_input_ids": torch.randint(10, (batch_size, 500), dtype=torch.int64),
                "diff_attention_mask": torch.ones(batch_size, 500, dtype=torch.int64),
                "msg_input_ids": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
                "msg_attention_mask": torch.ones(batch_size, self.max_len, dtype=torch.int64),
                "msg_labels": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
                "generation_input_ids": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
                "generation_attention_mask": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
            }
