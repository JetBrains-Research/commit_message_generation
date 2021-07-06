import torch
from transformers import GPT2LMHeadModel, BeamSearchScorer  # type: ignore
from transformers.generation_utils import GenerationMixin  # type: ignore
from transformers.file_utils import ModelOutput  # type: ignore
from typing import Optional, Dict, List, Callable


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
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        min_length: Optional[int] = None,
        length_penalty: Optional[float] = None,
        max_length: Optional[int] = 20,
        num_beams: Optional[int] = 1,
        do_early_stopping: Optional[bool] = None,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = None,
        output_scores: Optional[bool] = True,
        return_dict_in_generate: Optional[bool] = True,
    ):
        # set init values
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        do_early_stopping = do_early_stopping if do_early_stopping is not None else self.config.early_stopping

        if num_beam_groups > num_beams:  # type: ignore
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        if num_beam_hyps_to_keep > num_beams:  # type: ignore
            raise ValueError("`num_beam_hyps_to_keep` has to be smaller or equal to `num_beams`")

        logits_processors_list = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=self.config.eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        beam_search_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,
            num_beam_hyps_to_keep=num_beam_hyps_to_keep,
            num_beam_groups=num_beam_groups,
        )

        is_encoder_decoder = encoder_outputs is not None

        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            expand_size=num_beams,
            is_encoder_decoder=is_encoder_decoder,
        )

        encoder_hidden_states = model_kwargs["encoder_outputs"][0] if encoder_outputs is not None else None
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:2], device=self.device)

        return self.beam_search(
            input_ids=input_ids,
            attention_mask=model_kwargs["attention_mask"],
            beam_scorer=beam_search_scorer,
            logits_processor=logits_processors_list,
            max_length=max_length,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
