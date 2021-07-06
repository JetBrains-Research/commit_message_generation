import os
import wandb
import hydra
from omegaconf import DictConfig
from pathlib import Path
from onnxruntime.transformers import optimizer
from transformers.convert_graph_to_onnx import convert
from transformers import AutoConfig
from model.encoder_decoder_module import EncoderDecoderModule


class ONNXConverter:
    def __init__(self, **kwargs):
        self.hparams = kwargs
        self.hparams["paths"] = {key: os.path.join(hydra.utils.get_original_cwd(), value)
                                  for key, value in kwargs["paths"].items()}

    def convert_and_optimize(self):
        if not (os.path.exists(self.hparams["paths"]["seq2seq_model"]) and
                os.path.isdir(self.hparams["paths"]["seq2seq_model"])) \
                or not os.listdir(self.hparams["paths"]["seq2seq_model"]):
            self.get_wandb_artifact()

        if not (os.path.exists(self.hparams["paths"]["encoder"]) and os.path.isdir(self.hparams["paths"]["encoder"])) \
                or not os.listdir(self.hparams["paths"]["encoder"]):
            self.save_encoder_from_seq2seq_ckpt()

        if not Path(self.hparams["paths"]["onnx_encoder"]).is_file():
            self.convert_encoder_to_onnx()

        if not Path(self.hparams["paths"]["optimized_onnx_encoder"]).is_file():
            self.optimize_onnx()

    def get_wandb_artifact(self):
        api = wandb.Api()
        artifact = api.artifact(self.hparams["wandb_artifact"])
        os.makedirs(self.hparams["paths"]["seq2seq_model"], exist_ok=True)
        artifact.checkout(root=self.hparams["paths"]["seq2seq_model"])

    def save_encoder_from_seq2seq_ckpt(self):
        os.makedirs(self.hparams["paths"]["encoder"], exist_ok=True)
        model = EncoderDecoderModule.load_from_checkpoint(os.path.join(self.hparams["paths"]["seq2seq_model"],
                                                                       os.listdir(self.hparams["paths"]["seq2seq_model"])[0]),
                                                          num_gpus=1).model
        model.encoder.save_pretrained(save_directory=self.hparams["paths"]["encoder"])

    def convert_encoder_to_onnx(self):
        convert(framework="pt",
                model=self.hparams["paths"]["encoder"],
                tokenizer=self.hparams["tokenizer"],
                output=Path(self.hparams["paths"]["onnx_encoder"]),
                opset=self.hparams["opset"])

    def optimize_onnx(self):
        config = AutoConfig.from_pretrained(self.hparams["paths"]["encoder"])
        optimized_model = optimizer.optimize_model(self.hparams["paths"]["onnx_encoder"],
                                                   model_type="bert",
                                                   num_heads=config.num_attention_heads,
                                                   hidden_size=config.hidden_size)
        optimized_model.convert_model_float32_to_float16()
        optimized_model.save_model_to_file(self.hparams["paths"]["optimized_onnx_encoder"])
