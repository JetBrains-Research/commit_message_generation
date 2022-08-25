from typing import List

from omegaconf import DictConfig


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
    def _get_model_tags(model_cfg: DictConfig) -> List[str]:
        tags = []

        tags.append(model_cfg.model_configuration)

        for model_part in ["encoder", "decoder"]:
            if model_part == "encoder" and model_cfg.model_configuration == "decoder":
                continue
            if f"{model_part}_name_or_path" in model_cfg:
                tags.append(
                    f"{model_part}: {WandbOrganizer._prepare_pretrained_name(model_cfg[f'{model_part}_name_or_path'])}"
                )
            else:
                tags.append(f"{model_part}: random_{model_cfg[f'{model_part}_model_type']}")
            if f"num_layers_{model_part}" in model_cfg:
                tags[-1] += f" {model_cfg[f'num_layers_{model_part}']} layers"

        if model_cfg.model_configuration == "encoder_decoder":
            if model_cfg.tie_encoder_decoder:
                tags.append("shared weights")
            elif model_cfg.tie_word_embeddings:
                tags.append("shared embeddings")

        return tags

    @staticmethod
    def get_run_name(model_cfg: DictConfig, dataset_cfg: DictConfig) -> str:
        name = []
        for model_part in ["encoder", "decoder"]:
            if model_part == "encoder" and model_cfg.model_configuration == "decoder":
                continue
            if f"{model_part}_name_or_path" in model_cfg:
                name.append(f"{WandbOrganizer._prepare_pretrained_name(model_cfg[f'{model_part}_name_or_path'])}")
            else:
                name.append(f"random_{model_cfg[f'{model_part}_model_type']}")

            if f"num_layers_{model_part}" in model_cfg:
                name.append(f"{model_cfg[f'num_layers_{model_part}']}")

        if model_cfg.model_configuration == "encoder_decoder":
            if model_cfg.tie_encoder_decoder:
                name.append("shared")
            elif model_cfg.tie_word_embeddings:
                name.append("shared-embeddings")
        name.append("with-history" if dataset_cfg.train_with_history else "without-history")

        return "_".join(name)

    @staticmethod
    def get_tags_train(model_cfg: DictConfig, dataset_cfg: DictConfig) -> List[str]:
        tags = WandbOrganizer._get_model_tags(model_cfg)
        tags.append("train with history" if dataset_cfg.train_with_history else "train without history")
        return tags

    @staticmethod
    def get_tags_generate(dataset_cfg: DictConfig) -> List[str]:
        tags = [
            "generate with history" if dataset_cfg.generate_with_history else "generate without history",
            f"context ratio = {dataset_cfg.context_ratio}",
        ]
        return tags
