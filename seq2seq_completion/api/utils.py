import torch
from seq2seq_completion.model import EncoderDecoder
from seq2seq_completion.data_utils import DataProcessor
from seq2seq_completion.api.aws_utils import load_aws_model


def create_model(encoder_name_or_path: str, decoder_name_or_path: str, device=None) -> EncoderDecoder:
    load_aws_model(encoder_name_or_path=encoder_name_or_path, decoder_name_or_path=decoder_name_or_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(encoder_name_or_path=encoder_name_or_path, decoder_name_or_path=decoder_name_or_path)
    model.to(device)
    return model


def create_processor(
    prompt_max_len: int, diff_tokenizer_name_or_path: str, msg_tokenizer_name_or_path: str, preprocessing: bool = True
) -> DataProcessor:
    processor = DataProcessor(
        prompt_max_len=prompt_max_len,
        diff_tokenizer_name_or_path=diff_tokenizer_name_or_path,
        msg_tokenizer_name_or_path=msg_tokenizer_name_or_path,
        preprocessing=preprocessing,
    )
    return processor
