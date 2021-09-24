from typing import List, Optional
from transformers.file_utils import ModelOutput
from seq2seq_completion.model import EncoderDecoder
from seq2seq_completion.data_utils import DataProcessor
from seq2seq_completion.api.setup_utils import create_model, create_processor


class ServerCMCApi:
    _model: EncoderDecoder
    _processor: DataProcessor

    @staticmethod
    def setup(
        encoder_name_or_path: str,
        decoder_name_or_path: str,
        prompt_max_len: int,
        preprocessing: bool = True,
        device=None,
    ):
        ServerCMCApi._model = create_model(encoder_name_or_path, decoder_name_or_path, device)

        ServerCMCApi._processor = create_processor(
            prompt_max_len,
            encoder_name_or_path,
            decoder_name_or_path,
            preprocessing,
        )

    @staticmethod
    def complete(
        decoder_context: str,
        prefix: Optional[str] = None,
        diff: Optional[str] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        min_length: int = 5,
        max_length: int = 5,
        num_beams: int = 4,
        num_return_sequences: int = 4,
        **generation_kwargs,
    ) -> List[str]:
        # prepare input for generation
        model_input = ServerCMCApi._processor(decoder_context=decoder_context, diff=diff)

        # generate
        results = ServerCMCApi._model.generate(
            input_ids=model_input["decoder_input_ids"].to(ServerCMCApi._model.decoder.device),
            encoder_input_ids=model_input["encoder_input_ids"].to(ServerCMCApi._model.decoder.device),
            encoder_outputs=encoder_outputs,
            prefix=prefix,
            tokenizer=ServerCMCApi._processor._msg_tokenizer,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            min_length=min_length + model_input["decoder_input_ids"].shape[1],
            max_length=max_length + model_input["decoder_input_ids"].shape[1],
            **generation_kwargs,
        )

        # remove prompt from generated tensors
        results["sequences"] = results["sequences"][:, model_input["decoder_input_ids"].shape[1] :]

        # decode generated sequences
        return ServerCMCApi._processor._msg_tokenizer.batch_decode(results["sequences"], skip_special_tokens=True)
