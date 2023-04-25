import logging
import os

import hydra
import openai
import wandb
from omegaconf import OmegaConf

from conf import OpenAIConfig
from src import DataPreprocessor, OpenAIUtils, TokenEstimator


@hydra.main(version_base="1.1", config_path="conf", config_name="openai_config")
def main(cfg: OpenAIConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    if cfg.logger.use_wandb:
        if cfg.logger.use_api_key:
            with open(hydra.utils.to_absolute_path("wandb_api_key.txt"), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()

        wandb.init(
            project=cfg.logger.project,
            name=f"{cfg.model_id}_{cfg.dataset.prompt_configuration}",
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
            job_type="openai",
        )

    with open(hydra.utils.to_absolute_path(os.environ["API_KEY_PATH"]), "r") as f:
        openai.api_key = f.read().strip()

    openai_model = OpenAIUtils(cfg.model_id, cfg.generation)
    token_estimator = TokenEstimator(cfg.model_id)

    use_chat = cfg.model_id in ["gpt-3.5-turbo", "gpt-4"]

    preprocessor = DataPreprocessor(
        tokenizer=token_estimator.tokenizer,
        max_number_of_tokens=cfg.dataset.max_number_of_tokens,
        prompt_configuration=cfg.dataset.prompt_configuration,
        use_chat=use_chat,
    )

    # -----------------------
    #    process prompts    -
    # -----------------------
    cfg.dataset.root_dir = hydra.utils.to_absolute_path(cfg.dataset.root_dir)
    processed_path = f"{cfg.dataset.root_dir}/{cfg.dataset.prompt_configuration}_{cfg.dataset.input_path}"
    preprocessor.process_file(
        input_path=f"{cfg.dataset.root_dir}/{cfg.dataset.input_path}",
        output_path=processed_path,
        chunksize=cfg.dataset.chunksize,
        use_cache=cfg.dataset.use_cache,
    )

    # -----------------------
    #     estimate tokens   -
    # -----------------------
    if cfg.fill_file:
        logging.info("Configured to fill unfinished predictions file.")
        assert cfg.dataset.input_path_unfinished_preds is not None
        cfg.dataset.input_path_unfinished_preds = hydra.utils.to_absolute_path(cfg.dataset.input_path_unfinished_preds)
        num_tokens_prompts, num_tokens_completion = token_estimator.get_num_tokens_unfinished_file(
            input_path_prompts=processed_path,
            input_path_predictions=cfg.dataset.input_path_unfinished_preds,
            num_tokens_to_generate=cfg.generation.max_tokens,
        )
    else:
        num_tokens_prompts, num_tokens_completion = token_estimator.get_num_tokens_file(
            processed_path, num_tokens_to_generate=cfg.generation.max_tokens
        )

    logging.warning(
        f"Expected to consume {num_tokens_prompts + num_tokens_completion} tokens. "
        f"Currently, it would be {(num_tokens_prompts + num_tokens_completion) / 1000 * 0.002:.2f}$ for ChatGPT "
        f"and {(num_tokens_prompts / 1000 * 0.03) + (num_tokens_completion / 1000 * 0.06):.2f}$ for GPT-4."
    )
    answer = input(f"Configured to use {cfg.model_id} model. Do you want to continue? (y/n): ")
    if answer.lower() not in ["yes", "y"]:
        logging.warning("Decided not to proceed.")
        return

    # ----------------------
    #       query API      -
    # ----------------------
    output_path = f"{cfg.model_id}_{cfg.dataset.prompt_configuration}_{cfg.dataset.input_path}"

    if cfg.fill_file:
        logging.info("Configured to fill unfinished predictions file.")
        assert cfg.dataset.input_path_unfinished_preds is not None
        assert use_chat, "Currently, filling unfinished predictions file is not supported for Completion endpoint."
        openai_model.fill_completion_chat_file(
            input_path_prompts=processed_path,
            input_path_predictions=cfg.dataset.input_path_unfinished_preds,
            output_path=output_path,
        )
    else:
        if use_chat:
            openai_model.get_completion_chat_file(processed_path, output_path)
        else:
            openai_model.get_completion_file(processed_path, output_path)

    # -------------------------------------------------
    #       upload predictions to W&B (optional)      -
    # -------------------------------------------------
    if cfg.logger.use_wandb and cfg.logger.upload_artifact:
        artifact = wandb.Artifact(
            name=f"{cfg.model_id}_{cfg.dataset.prompt_configuration}" + "_preds",
            type="openai_preds",
        )
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
