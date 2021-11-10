from dataclasses import dataclass
from typing import List, Union, Dict

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class NextTokenPredictionCollator:
    """
    Data collator used to
    1) collate author's history with current message (optional)
    2) pad and concatenate everything into tensors
    - diff_input_ids, diff_attention_mask: encoder input (simply diffs)
    - msg_input_ids, msg_attention_mask, msg_labels: decoder input for next token prediction metrics
        - <bos> history_1 \n ... history_k \n cur_msg <eos> (history + current message)
    """

    src_tokenizer: PreTrainedTokenizerBase
    trg_tokenizer: PreTrainedTokenizerBase
    max_len: int = 200
    with_history: bool = True

    def __call__(
        self, examples: List[Dict[str, Union[List[List[int]], List[List[List[int]]]]]]
    ) -> Dict[str, torch.Tensor]:
        diff_inputs = [e["diff_input_ids"] for e in examples]  # 2D - list of lists
        message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists
        history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists

        all_msg_ids = []  # input for training or metrics: history + cur_msg (right-side padding)
        all_msg_masks = []  # 0 on pad tokens and 1 otherwise (right-side padding)
        all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)

        # concatenate history examples with current input ids (checking that resulting length is <= max_len)
        for i, message_ids in enumerate(message_inputs):
            cur_ids = [
                [self.trg_tokenizer.bos_token_id],
                message_ids[: self.max_len - 2],
                [self.trg_tokenizer.eos_token_id],
            ]
            cur_labels = [[-100], message_ids[: self.max_len - 2], [-100]]
            cur_len = len(message_ids[: self.max_len - 2]) + 2

            if self.with_history:
                for history_input_ids in history_inputs[i][::-1]:
                    # insert prev messages from history until we reach max_len
                    if cur_len + len(history_input_ids) + len(self.trg_tokenizer(r" \n ").input_ids) > self.max_len:
                        break

                    cur_len += len(history_input_ids) + len(self.trg_tokenizer(r" \n ").input_ids)

                    cur_ids.insert(1, history_input_ids + self.trg_tokenizer(r" \n ").input_ids)
                    cur_labels.insert(1, [-100 for _ in history_input_ids + self.trg_tokenizer(r" \n ").input_ids])

            # flatten everything into one sequence and convert to tensor of torch.int64
            cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)

            # create ones for attention mask
            cur_mask = torch.ones_like(cur_ids)

            all_msg_ids.append(cur_ids)
            all_msg_labels.append(cur_labels)
            all_msg_masks.append(cur_mask)

        all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
        all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

        input_max_len = max(len(tensor) for tensor in all_msg_ids)
        diff_max_len = max(len(tensor) for tensor in all_diff_ids)

        # pad tensors to max length in batch
        for i, (diff_id_tensor, diff_mask_tensor, msg_id_tensor, msg_mask_tensor, msg_labels_tensor,) in enumerate(
            zip(
                all_diff_ids,
                all_diff_masks,
                all_msg_ids,
                all_msg_masks,
                all_msg_labels,
            )
        ):
            # pad ids with pad_token_id (which doesn't really matter for GPT-2)
            all_msg_ids[i] = torch.nn.functional.pad(
                msg_id_tensor,
                pad=[0, input_max_len - msg_id_tensor.numel()],
                mode="constant",
                value=self.trg_tokenizer.pad_token_id,
            )

            all_diff_ids[i] = torch.nn.functional.pad(
                diff_id_tensor,
                pad=[0, diff_max_len - diff_id_tensor.numel()],
                mode="constant",
                value=self.src_tokenizer.pad_token_id,
            )

            # pad labels with -100
            all_msg_labels[i] = torch.nn.functional.pad(
                msg_labels_tensor, pad=[0, input_max_len - msg_labels_tensor.numel()], mode="constant", value=-100
            )

            # pad masks with zeros
            all_msg_masks[i] = torch.nn.functional.pad(
                msg_mask_tensor, pad=[0, input_max_len - msg_mask_tensor.numel()], mode="constant", value=0
            )
            all_diff_masks[i] = torch.nn.functional.pad(
                diff_mask_tensor, pad=[0, diff_max_len - diff_mask_tensor.numel()], mode="constant", value=0
            )

        all_msg_ids = torch.stack(all_msg_ids)
        all_msg_masks = torch.stack(all_msg_masks)
        all_msg_labels = torch.stack(all_msg_labels)
        all_diff_ids = torch.stack(all_diff_ids)
        all_diff_masks = torch.stack(all_diff_masks)

        return {
            "diff_input_ids": all_diff_ids,
            "diff_attention_mask": all_diff_masks,
            "msg_input_ids": all_msg_ids,
            "msg_attention_mask": all_msg_masks,
            "msg_labels": all_msg_labels,
        }


@dataclass
class GenerationCollator:
    """
    Data collator used to
    1) collate author's history with current message (optional)
    2) add first X% characters of each message to generation context
    3) pad and concatenate everything into tensor
    - diff_input_ids, diff_attention_mask: encoder input (simply diffs)
    - msg_input_ids, msg_attention_mask: generation context
        - <bos> history_1 \n ... history_k \n start of cur_msg (without last word) <eos>
    - target: message without first X% characters, used for metrics computation (string)
    - prefix: last word from context, used for prefix-contrained generation (string)
    """

    src_tokenizer: PreTrainedTokenizerBase
    trg_tokenizer: PreTrainedTokenizerBase
    max_len: int = 200
    with_history: bool = True
    context_ratio: float = 1.0

    def __call__(
        self, examples: List[Dict[str, Union[List[List[int]], List[List[List[int]]]]]]
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        diff_inputs = [e["diff_input_ids"] for e in examples]  # 2D - list of lists
        message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists
        history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists

        all_msg_ids = []  # 'prompt' for generation: history (left-side padding)
        all_msg_masks = []  # 0 on pad tokens and 1 otherwise (left-side padding)
        all_msg_targets = []  # targets for generation (right-side padding)
        all_prefixes = []  # make sure to separate last word from context to use for prefix-constrained generation

        for i, message_ids in enumerate(message_inputs):
            message_ids = message_ids[: self.max_len - 1]

            decoded_message = self.trg_tokenizer.decode(message_ids, skip_special_tokens=True)
            context_len = len(decoded_message) * self.context_ratio
            decoded_input, decoded_target = decoded_message[: int(context_len)], decoded_message[int(context_len) :]
            try:
                decoded_input, decoded_prefix = " ".join(decoded_input.split()[:-1]), decoded_input.split()[-1]
                if len(decoded_input) > 0:
                    decoded_prefix = " " + decoded_prefix
                decoded_target = decoded_prefix + decoded_target

                tokenized_input = self.trg_tokenizer(decoded_input).input_ids
                tokenized_target = self.trg_tokenizer(decoded_target, return_tensors="pt").input_ids.squeeze()
            except IndexError:
                print(
                    f"Target '{decoded_target}' is too short: {len(decoded_target)} * {self.context_ratio} = {context_len}"
                )
                print()

                tokenized_input = []
                decoded_prefix = ""
                tokenized_target = self.trg_tokenizer(decoded_target, return_tensors="pt").input_ids.squeeze()

            cur_ids = [[self.trg_tokenizer.bos_token_id], tokenized_input]
            cur_len = len(tokenized_input) + 1

            if self.with_history:
                for history_input_ids in history_inputs[i][::-1]:
                    # insert prev messages from history until we reach max_len
                    if cur_len + len(history_input_ids) + len(self.trg_tokenizer._sep) > self.max_len:
                        break

                    cur_len += len(history_input_ids) + len(self.trg_tokenizer._sep)
                    cur_ids.insert(1, history_input_ids + self.trg_tokenizer._sep)

            # flatten everything into one sequence and convert to tensor of torch.int64
            cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)

            # create ones for attention mask
            cur_mask = torch.ones_like(cur_ids)

            all_msg_ids.append(cur_ids)
            all_msg_masks.append(cur_mask)
            all_prefixes.append(decoded_prefix)
            all_msg_targets.append(tokenized_target)

        all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
        all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

        input_max_len = max(len(tensor) for tensor in all_msg_ids)
        target_max_len = max(len(tensor) for tensor in all_msg_targets)
        diff_max_len = max(len(tensor) for tensor in all_diff_ids)

        # pad tensors to max length in batch
        # NOTE: left side padding for generation!! https://github.com/huggingface/transformers/issues/3021
        for i, (diff_id_tensor, diff_mask_tensor, msg_id_tensor, msg_mask_tensor, msg_target_tensor) in enumerate(
            zip(all_diff_ids, all_diff_masks, all_msg_ids, all_msg_masks, all_msg_targets)
        ):
            # pad ids with pad_token_id (which doesn't really matter for GPT-2)
            all_msg_ids[i] = torch.nn.functional.pad(
                msg_id_tensor,
                pad=[input_max_len - msg_id_tensor.numel(), 0],
                mode="constant",
                value=self.trg_tokenizer.pad_token_id,
            )
            all_diff_ids[i] = torch.nn.functional.pad(
                diff_id_tensor,
                pad=[0, diff_max_len - diff_id_tensor.numel()],
                mode="constant",
                value=self.src_tokenizer.pad_token_id,
            )
            all_msg_targets[i] = torch.nn.functional.pad(
                msg_target_tensor,
                pad=[0, target_max_len - msg_target_tensor.numel()],
                mode="constant",
                value=self.trg_tokenizer.pad_token_id,
            )

            # pad masks with zeros
            all_msg_masks[i] = torch.nn.functional.pad(
                msg_mask_tensor, pad=[input_max_len - msg_mask_tensor.numel(), 0], mode="constant", value=0
            )
            all_diff_masks[i] = torch.nn.functional.pad(
                diff_mask_tensor, pad=[0, diff_max_len - diff_mask_tensor.numel()], mode="constant", value=0
            )

        all_msg_ids = torch.stack(all_msg_ids)
        all_msg_masks = torch.stack(all_msg_masks)
        all_msg_targets = torch.stack(all_msg_targets)
        all_diff_ids = torch.stack(all_diff_ids)
        all_diff_masks = torch.stack(all_diff_masks)

        return {
            "diff_input_ids": all_diff_ids,
            "diff_attention_mask": all_diff_masks,
            "msg_input_ids": all_msg_ids,
            "msg_attention_mask": all_msg_masks,
            "target": all_msg_targets,
            "prefix": all_prefixes,
        }
