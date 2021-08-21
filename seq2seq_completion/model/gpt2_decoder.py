import torch
from transformers import GPT2LMHeadModel, BeamSearchScorer, PreTrainedTokenizerBase
from transformers.generation_utils import GenerationMixin
from transformers.file_utils import ModelOutput
from typing import Optional, Dict, List

from seq2seq_completion.model.prefix_utils import PrefixAllowedTokens


class GPT2Decoder(GPT2LMHeadModel):
    """
    Wrapper class for GPT2LMHeadModel to support processing encoder outputs during decoder-only generation.
    """

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        This method is called inside beam_search.
        Default version from GPT2LMHeadModel ignores encoder outputs even if they are present in **kwargs.
        This version simply adds encoder outputs to resulting dictionary.
        """
        model_inputs = super(GPT2Decoder, self).prepare_inputs_for_generation(input_ids, past, **kwargs)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        model_inputs.update(
            {"encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": encoder_attention_mask}
        )
        return model_inputs

    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        prefix: Optional[str] = None,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        min_length: Optional[int] = None,
        max_length: Optional[int] = 20,
        early_stopping: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        diversity_penalty: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences conditioned on provided inputs via beam search.

        Note: signature is almost the same as in `generate` method from transformers,
        see `transformers.generation_utils.GenerationMixin.generate` for more information.

        :return: dictionary (with keys `sequences` and `scores`)
        """
        # use default values from model config for several parameters if they are not passed explicitly
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_beam_groups > num_beams:  # type: ignore
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        if num_return_sequences > num_beams:  # type: ignore
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`")

        prefix_allowed_tokens_fn = (
            PrefixAllowedTokens(prefix=prefix, context_len=input_ids.shape[1], tokenizer=tokenizer)
            if prefix is not None
            else None
        )

        # get list of logits processors according to parameters (reusing method from transformers)
        logits_processors_list = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=tokenizer("\n").input_ids[0],
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # get beam search scorer according to parameters
        beam_search_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )

        # make tensor of ones for attention mask if it is not passed explicitly
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        # expand inputs for running beam search (1st dimension is multiplied by num_beams)
        is_encoder_decoder = encoder_outputs is not None
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            expand_size=num_beams,
            is_encoder_decoder=is_encoder_decoder,
        )

        # make tensor of ones for encoder attention mask if it is not passed explicitly
        encoder_hidden_states = model_kwargs["encoder_outputs"][0] if encoder_outputs is not None else None
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:2], device=self.device)

        # run beam search
        beam_search_output = self.beam_search(
            input_ids=input_ids,
            attention_mask=model_kwargs["attention_mask"],
            beam_scorer=beam_search_scorer,
            logits_processor=logits_processors_list,
            max_length=max_length,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=tokenizer("\n").input_ids[0],
            output_scores=True,
            return_dict_in_generate=True,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return {"sequences": beam_search_output.sequences, "scores": beam_search_output.sequences_scores}
