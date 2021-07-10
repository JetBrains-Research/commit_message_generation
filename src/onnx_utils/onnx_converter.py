import os
import wandb
from typing import Union
from pathlib import Path
from onnxruntime.transformers import optimizer  # type: ignore
from transformers.convert_graph_to_onnx import convert, quantize  # type: ignore
from transformers import AutoConfig  # type: ignore
from src.model_utils import EncoderDecoderModule


class ONNXConverter:
    def __init__(
        self,
        wandb_artifact: str,
        tokenizer: str,
        opset: int,
        seq2seq_model_path: Union[str, os.PathLike],
        encoder_path: Union[str, os.PathLike],
        onnx_encoder_path: Union[str, os.PathLike],
        optimized_onnx_encoder_path: Union[str, os.PathLike],
    ):
        """
        Class for converting encoder from seq2seq model to optimized ONNX.

        :param wandb_artifact: name of wandb artifact (in format `project/artifact:alias`)
        :param tokenizer: name of model tokenizer on huggingface hub
        :param opset: which ONNX opset to use
        :param seq2seq_model_path: path to folder with seq2seq model weights
        :param encoder_path: path to folder with encoder weights
        :param onnx_encoder_path: path to converted .onnx encoder file
        :param optimized_onnx_encoder_path: path to optimized .onnx encoder file
        """
        self.wandb_artifact = wandb_artifact
        self.opset = opset
        self.tokenizer = tokenizer
        self.seq2seq_model_path = seq2seq_model_path
        self.encoder_path = encoder_path
        self.onnx_encoder_path = onnx_encoder_path
        self.optimized_onnx_encoder_path = optimized_onnx_encoder_path

    def convert_and_optimize(self) -> Path:
        """
        Performs all possibly necessary steps:
            1) Downloads seq2seq model weights from wandb
            2) Saves encoder from seq2seq model separately
            3) Converts encoder to ONNX
            4) Optimizes ONNX graph
            5) Quantizes ONNX graph
        """
        if not (os.path.exists(self.seq2seq_model_path) and os.path.isdir(self.seq2seq_model_path)) or not os.listdir(
            self.seq2seq_model_path
        ):  # if folder doesn't exist or is empty
            ONNXConverter.get_wandb_artifact(
                wandb_artifact=self.wandb_artifact, seq2seq_model_path=self.seq2seq_model_path
            )

        if not (os.path.exists(self.encoder_path) and os.path.isdir(self.encoder_path)) or not os.listdir(
            self.encoder_path
        ):  # if folder doesn't exist or is empty
            ONNXConverter.save_encoder_from_seq2seq_ckpt(
                seq2seq_model_path=self.seq2seq_model_path, encoder_path=self.encoder_path
            )

        if not Path(self.onnx_encoder_path).is_file():  # if file doesn't exist
            ONNXConverter.convert_encoder_to_onnx(
                encoder_path=self.encoder_path,
                onnx_encoder_path=self.onnx_encoder_path,
                tokenizer=self.tokenizer,
                opset=self.opset,
            )

        if not Path(self.optimized_onnx_encoder_path).is_file():  # if file doesn't exist
            ONNXConverter.optimize_onnx(
                encoder_path=self.encoder_path,
                onnx_encoder_path=self.onnx_encoder_path,
                optimized_onnx_encoder_path=self.optimized_onnx_encoder_path,
            )

        return ONNXConverter.quantize_onnx(Path(self.optimized_onnx_encoder_path))

    @staticmethod
    def get_wandb_artifact(wandb_artifact: str, seq2seq_model_path: Union[str, os.PathLike]):
        """
        Loads artifact from Weights & Biases.

        :param wandb_artifact: name of wandb artifact (in format `project/artifact:alias`)
        :param seq2seq_model_path: path to folder to save seq2seq model weights
        """
        api = wandb.Api()
        artifact = api.artifact(wandb_artifact)
        os.makedirs(seq2seq_model_path, exist_ok=True)
        artifact.checkout(seq2seq_model_path)

    @staticmethod
    def save_encoder_from_seq2seq_ckpt(
        seq2seq_model_path: Union[str, os.PathLike], encoder_path: Union[str, os.PathLike]
    ):
        """
        Loads seq2seq model weights from checkpoint and saves encoder separately.

        :param seq2seq_model_path: path to folder to load seq2seq model weights from
        :param encoder_path: path to folder to save encoder weights in
        """
        os.makedirs(encoder_path, exist_ok=True)
        model = EncoderDecoderModule.load_from_checkpoint(
            os.path.join(seq2seq_model_path, os.listdir(seq2seq_model_path)[0]), num_gpus=1
        ).model
        model.encoder.save_pretrained(save_directory=encoder_path)

    @staticmethod
    def convert_encoder_to_onnx(
        encoder_path: Union[str, os.PathLike], onnx_encoder_path: Union[str, os.PathLike], tokenizer: str, opset: int
    ):
        """
        Converts encoder to onnx (via script from transformers).

        :param encoder_path: path to folder to load encoder weights from
        :param onnx_encoder_path: path to converted .onnx encoder file
        :param tokenizer: name of model tokenizer on huggingface hub
        :param opset: which ONNX opset to use
        """
        convert(framework="pt", model=encoder_path, tokenizer=tokenizer, output=Path(onnx_encoder_path), opset=opset)

    @staticmethod
    def quantize_onnx(onnx_encoder_path: Path) -> Path:
        """
        Quantizes onnx encoder (via script from transformers).

        :param onnx_encoder_path: path to converted .onnx encoder file
        :returns: generated path to quantized model
        """
        return quantize(onnx_encoder_path)

    @staticmethod
    def optimize_onnx(
        encoder_path: Union[str, os.PathLike],
        onnx_encoder_path: Union[str, os.PathLike],
        optimized_onnx_encoder_path: Union[str, os.PathLike],
    ):
        """
        Optimizes onnx graph (via script from onnxruntime).

        :param encoder_path: path to folder to load encoder config from
        :param onnx_encoder_path: path to converted .onnx encoder file
        :param optimized_onnx_encoder_path: path to optimized .onnx encoder file
        """
        config = AutoConfig.from_pretrained(encoder_path)
        optimized_model = optimizer.optimize_model(
            onnx_encoder_path, model_type="bert", num_heads=config.num_attention_heads, hidden_size=config.hidden_size
        )
        optimized_model.save_model_to_file(optimized_onnx_encoder_path)
