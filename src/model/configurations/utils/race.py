from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class RACE(T5ForConditionalGeneration):
    """
    This class implements RACE model as a wrapper over T5ForConditionalGeneration
    that is able to utilize all generation functionality from transformers.

    It is based on paper "RACE: Retrieval-Augmented Commit Message Generation" from EMNLP 2022
    and its replication package – https://github.com/DeepSoftwareAnalytics/RACE.
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.W_s = nn.Linear(2 * config.d_model, 1)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Copied from `transformers.generation_utils` with a small change: when retrieved examples
        are present in model_kwargs, make sure to use custom encoder logic.

        This tweak is required for `generate` method from transformers to work correctly.
        """

        # 1. get encoder
        encoder = self.get_encoder()  # type: ignore[attr-defined]

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name  # type: ignore[attr-defined]
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # --------------------
        # modification starts
        if "retrieved_diff_input_ids" in model_kwargs and "retrieved_msg_input_ids" in model_kwargs:
            encoder_outputs, attention_mask = self._prepare_encoder_outputs(**encoder_kwargs)
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["attention_mask"] = attention_mask
        else:
            model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)
        # modification ends
        # ------------------

        return model_kwargs

    def _prepare_encoder_outputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        retrieved_diff_input_ids: Optional[torch.LongTensor] = None,
        retrieved_diff_attention_mask: Optional[torch.LongTensor] = None,
        retrieved_msg_input_ids: Optional[torch.LongTensor] = None,
        retrieved_msg_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[BaseModelOutput, torch.Tensor]:
        """
        Runs custom encoder logic that utilizes retrieved examples.

        Args:
            input_ids: Input ids for source sequence (diff).
            attention_mask: Attention mask for source sequence (diff).
            retrieved_diff_input_ids: Input ids for source sequence (diff) from retrieved example.
            retrieved_diff_attention_mask: Attention mask for source sequence (diff) from retrieved example.
            retrieved_msg_input_ids: Input ids for target sequence (message) from retrieved example.
            retrieved_msg_attention_mask: Attention mask for target sequence (message) from retrieved example.
            **kwargs: any keyword arguments for encoder

        Returns:
            A tuple of two elements, where first element is encoder output and the second is attention mask.
        """
        # Obtain input diff embedding
        # --------------------------------
        # (batch_size, diff_sequence_length, hidden_size)
        diff_last_hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # type: ignore[attr-defined]
        # (batch_size, hidden_size)
        diff_embedding = torch.mean(diff_last_hidden_state, dim=1)

        # Obtain retrieved diff embedding
        # --------------------------------
        # (batch_size, retrieved_diff_sequence_length, hidden_size)
        retrieved_diff_last_hidden_state = self.encoder(  # type: ignore[attr-defined]
            input_ids=retrieved_diff_input_ids, attention_mask=retrieved_diff_attention_mask
        ).last_hidden_state
        # (batch_size, hidden_size)
        retrieved_diff_embedding = torch.mean(retrieved_diff_last_hidden_state, dim=1)

        # Calculate similarity coefficient
        # --------------------------------
        # (batch_size, 2 * hidden_size)
        combined_embedding = torch.cat((diff_embedding, retrieved_diff_embedding), dim=1)
        # (batch_size, 1, 1)
        lam = torch.sigmoid(self.W_s(combined_embedding)).unsqueeze(-1)

        # Process retrieved message
        # --------------------------------
        # (batch_size, retrieved_msg_sequence_length, hidden_size)
        retrieved_msg_last_hidden_state = self.encoder(  # type: ignore[attr-defined]
            input_ids=retrieved_msg_input_ids, attention_mask=retrieved_msg_attention_mask
        ).last_hidden_state
        # (batch_size, diff_sequence_length + retrieved_msg_sequence_length, hidden_size)
        result_hidden_state = torch.cat(
            ((1 - lam) * diff_last_hidden_state, lam * retrieved_msg_last_hidden_state), dim=1
        )

        # Tweak return type and attn mask
        # --------------------------------
        result_encoder_outputs = BaseModelOutput(
            last_hidden_state=result_hidden_state, hidden_states=None, attentions=None  # type: ignore[arg-type]
        )
        # (batch_size,  diff_sequence_length + retrieved_msg_sequence_length)
        result_attention_mask = torch.cat((attention_mask, retrieved_msg_attention_mask), dim=1)  # type: ignore[arg-type]
        return result_encoder_outputs, result_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        retrieved_diff_input_ids: Optional[torch.LongTensor] = None,
        retrieved_diff_attention_mask: Optional[torch.LongTensor] = None,
        retrieved_msg_input_ids: Optional[torch.LongTensor] = None,
        retrieved_msg_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        A wrapper over T5ForConditionalGeneration.forward().

        If retrieved examples are passed, it runs custom encoder logic and passes encoder outputs to
        T5ForConditionalGeneration.forward().

        If retrieved examples are not passed, it simply runs T5ForConditionalGeneration.forward().
        """

        if not encoder_outputs and retrieved_diff_input_ids is not None and retrieved_msg_input_ids is not None:
            assert input_ids is not None

            # initialize attention masks to ones if they aren't passed explicitly
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)  # type: ignore[assignment]
            if retrieved_diff_attention_mask is None:
                retrieved_diff_attention_mask = torch.ones_like(retrieved_diff_input_ids)  # type: ignore[assignment]
            if retrieved_msg_attention_mask is None:
                retrieved_msg_attention_mask = torch.ones_like(retrieved_msg_input_ids)  # type: ignore[assignment]

            encoder_outputs, attention_mask = self._prepare_encoder_outputs(  # type: ignore[assignment]
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieved_diff_input_ids=retrieved_diff_input_ids,
                retrieved_diff_attention_mask=retrieved_diff_attention_mask,
                retrieved_msg_input_ids=retrieved_msg_input_ids,
                retrieved_msg_attention_mask=retrieved_msg_attention_mask,
            )

        return super().forward(  # type: ignore[misc]
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
