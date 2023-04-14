from typing import List

from conf import (
    BaseDecoderConfig,
    BaseEncoderDecoderConfig,
    BaseModelConfig,
    BaseRACEConfig,
    BaseSeq2SeqConfig,
)


class WandbOrganizer:
    @staticmethod
    def _prepare_pretrained_name(name_or_path: str) -> str:
        name_or_path = name_or_path.split("/")[-1]
        if "-" in name_or_path:
            name_or_path = name_or_path.split("-")[0]
        if "_" in name_or_path:
            name_or_path = name_or_path.split("_")[0]
        return name_or_path

    @staticmethod
    def _get_model_tags(model_cfg: BaseModelConfig) -> List[str]:
        tags = [model_cfg.configuration]

        if model_cfg.configuration == "seq2seq":
            model_cfg = BaseSeq2SeqConfig(**model_cfg)  # type: ignore[arg-type]
            tags.append(WandbOrganizer._prepare_pretrained_name(model_cfg.name_or_path))
        elif model_cfg.configuration == "race":
            model_cfg = BaseRACEConfig(**model_cfg)  # type: ignore[arg-type]
            tags.append(WandbOrganizer._prepare_pretrained_name(model_cfg.name_or_path))
        elif model_cfg.configuration == "decoder":
            model_cfg = BaseDecoderConfig(**model_cfg)  # type: ignore[arg-type]
            tags.append(WandbOrganizer._prepare_pretrained_name(model_cfg.decoder_name_or_path))
        elif model_cfg.configuration == "encoder_decoder":
            model_cfg = BaseEncoderDecoderConfig(**model_cfg)  # type: ignore[arg-type]

            if model_cfg.encoder_name_or_path:
                tags.append(f"[encoder]: {WandbOrganizer._prepare_pretrained_name(model_cfg.encoder_name_or_path)}")
            if model_cfg.encoder_model_type:
                tags.append(f"[encoder]: random_{model_cfg.encoder_model_type}")
            if model_cfg.num_layers_encoder:
                tags.append(f"[encoder]: {model_cfg.num_layers_encoder} layers")

            if model_cfg.decoder_name_or_path:
                tags.append(f"[decoder]: {WandbOrganizer._prepare_pretrained_name(model_cfg.decoder_name_or_path)}")
            if model_cfg.decoder_model_type:
                tags.append(f"[decoder]: random_{model_cfg.decoder_model_type}")
            if model_cfg.num_layers_decoder:
                tags.append(f"[decoder]: {model_cfg.num_layers_decoder} layers")

            if model_cfg.tie_encoder_decoder:
                tags.append("shared weights")
            elif model_cfg.tie_word_embeddings:
                tags.append("shared embeddings")

        return tags

    @staticmethod
    def get_run_name(model_cfg: BaseModelConfig, encoder_input_type: str, train_with_history: bool) -> str:
        name = []

        if model_cfg.configuration == "seq2seq":
            model_cfg = BaseSeq2SeqConfig(**model_cfg)  # type: ignore[arg-type]
            name.append(WandbOrganizer._prepare_pretrained_name(model_cfg.name_or_path))
        elif model_cfg.configuration == "race":
            model_cfg = BaseRACEConfig(**model_cfg)  # type: ignore[arg-type]
            name.append("race_" + WandbOrganizer._prepare_pretrained_name(model_cfg.name_or_path))
        elif model_cfg.configuration == "decoder":
            model_cfg = BaseDecoderConfig(**model_cfg)  # type: ignore[arg-type]
            name.append(WandbOrganizer._prepare_pretrained_name(model_cfg.decoder_name_or_path))
        elif model_cfg.configuration == "encoder_decoder":
            model_cfg = BaseEncoderDecoderConfig(**model_cfg)  # type: ignore[arg-type]
            if model_cfg.encoder_name_or_path:
                name.append(WandbOrganizer._prepare_pretrained_name(model_cfg.encoder_name_or_path))
            if model_cfg.encoder_model_type:
                name.append(f"random_{model_cfg.encoder_model_type}")
            if model_cfg.num_layers_encoder:
                name.append(str(model_cfg.num_layers_encoder))

            if model_cfg.decoder_name_or_path:
                name.append(WandbOrganizer._prepare_pretrained_name(model_cfg.decoder_name_or_path))
            if model_cfg.decoder_model_type:
                name.append(f"random_{model_cfg.decoder_model_type}")
            if model_cfg.num_layers_decoder:
                name.append(str(model_cfg.num_layers_decoder))

            if model_cfg.tie_encoder_decoder:
                name.append("shared-weights")
            elif model_cfg.tie_word_embeddings:
                name.append("shared-embeddings")

        if encoder_input_type == "diff":
            name.append("with-history" if train_with_history else "without-history")
        elif encoder_input_type == "history":
            name.append("history-input")

        return "_".join(name)

    @staticmethod
    def get_tags_train(model_cfg: BaseModelConfig, encoder_input_type: str, train_with_history: bool) -> List[str]:
        tags = WandbOrganizer._get_model_tags(model_cfg)
        tags.append("train with history" if train_with_history else "train without history")
        tags.append(encoder_input_type)
        return tags

    @staticmethod
    def get_tags_generate(generate_with_history: bool, context_ratio: float) -> List[str]:
        tags = [
            "generate with history" if generate_with_history else "generate without history",
            f"context ratio = {context_ratio}",
        ]
        return tags
