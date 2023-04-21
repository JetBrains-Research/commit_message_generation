# Commit message completion ~~(and generation)~~

![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge)

This repository contains code for training and evaluation of Transformer-based models for commit message completion
task.

## Requirements

* :snake: Python
* :floppy_disk: Dependencies
    * Neural networks frameworks: [PyTorch](https://pytorch.org/)
        , [🤗 Transformers](https://huggingface.co/transformers/)
        and [PyTorch Lightning](https://www.pytorchlightning.ai/)
    * Configuration: [Hydra](https://hydra.cc/)
    * Experiments tracking: [Weights & Biases](https://wandb.ai/site)
    * Metrics: [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
      , [🤗 Datasets](https://huggingface.co/docs/datasets/)
      and other packages necessary for specific metrics implementations
    * Lint & unit tests: [mypy](https://github.com/python/mypy), [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [pytest](https://docs.pytest.org/en/7.1.x/)

This project provides dependencies for two Python dependency managers:
* Poetry: [`poetry.lock`](poetry.lock), [`pyproject.toml`](pyproject.toml)
* pip: [`requirements.txt`](requirements.txt) (obtained through `poetry export --with dev,retrieval --output requirements.txt`)
     
## Usage

### Step 1: Prepare raw data

> :construction: At some point, we plan to publish our dataset of commits. Until then, and if you wish to utilize this project 
> for other data, refer to this section.

> :star2: The data for this project was obtained via [commits_dataset](https://github.com/saridormi/commits_dataset) repo. 

<details>
<summary>:yellow_heart: click here for more information on required data format</summary>

This project expects all dataset parts to be stored in a separate JSONLines files:
```
 ├── ...  # data directory
 │   ├── train.jsonl
 │   ├── val.jsonl
 │   └── test.jsonl
 └── ...
```

In our case, each input example is commit. Also note that commits from each author should be in chronological order. Specifically, the following keys are expected in each row:

* `author`: Unique identifier for the author of commit.
* `message`: Commit message.
* `mods`: A list of modification made in a commit. Each modification should contain the following keys:
  * `change_type`: Type of modification (string, one of `MODIFY`, `ADD`, `DELETE`, `RENAME`, `COPY`, `UNKNOWN`).
  * `old_path`: Path to file before the commit (`None` when `change_type` is `ADD`).
  * `new_path`: Path to file after the commit (`None` when `change_type` is `DELETE`).
  * `diff`: Output of the `git diff` command for this specific file.

</details>

### Step 2: Choose configuration

#### Model architecture

This project supports the following models:

* [Transformer Decoder](src/model/configurations/decoder_wrapper.py)
  * Refer to [:hugs: documentation for AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)
* [Transformer with pretrained encoders/decoders](src/model/configurations/encoder_decoder_wrapper.py)
  * Refer to [:hugs: documentation for EncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)
  
* [Pretrained Seq2Seq Transformer](src/model/configurations/seq2seq_wrapper.py)
  * Refer to [:hugs: documentation for AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM)

* [RACE](src/model/configurations/race_wrapper.py) 
  * [:scroll: RACE: Retrieval-Augmented Commit Message Generation](https://arxiv.org/abs/2203.02700v3)

For details refer to classes provided in [`src/model/configurations`](src/model/configurations) or base configs provided in [`conf/model/base_configs.py`](conf/model/base_configs.py).

You can find specific configs for the following models in [`conf/model/configs.py`](conf/model/configs.py):
* distilGPT-2
* randomly initialized Transformer
* CodeT5 from [:scroll: CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/abs/2109.00859)
* CodeReviewer from [:scroll: Automating Code Review Activities by Large-Scale Pre-training](https://arxiv.org/abs/2203.09095)
* RACE + T5 from [:scroll: RACE: Retrieval-Augmented Commit Message Generation](https://arxiv.org/abs/2203.02700v3)

#### Input type

This project explores two kinds of input for commit message completion task: diff and commit message history. 

* For decoder, there is only one supported option: concatenate commit message history with current commit message and pass to context.

* For seq2seq models, there are three supported options:
  * *Diff-only:* pass diff to encoder, pass current message to decoder.
  * *History-only:* pass history to encoder, pass current message to decoder.
  * *Diff + history:* pass diff to encoder, pass commit message history concatenated with current message to decoder.

### Step 3: Train

1. Define configuration for training at [`conf/train_config.py`](conf/train_config.py).
2. Choose one of available model configs or add your own.
3. Note that you have to define missing parameters from [`InputConfig`](conf/data/input_config.py). You can do it via CLI or just rewrite them. Below is the example how to define parameters via CLI.

To launch training of model defined as `XXXModelConfig` and registered via `ConfigStore.store(name="XXX", group="model", node=XXXModelConfig)`, run the following command (with actual values instead of X's):
```
python train.py +model=XXX ++input.train_with_history=X ++input.encoder_input_type=X
```

#### Additional steps for RACE model

Experiments with RACE model require a slightly different procedure.

1. Fine-tune CodeT5 model. Refer to the instruction above for details.

2. Use encoder from fine-tuned CodeT5 checkpoint to perform retrieval. 
   
    Define configuration in [`conf/retrieval_config.py`](conf/retrieval_config.py). You have to either provide a local path to checkpoint in `ckpt_path` or use W&B artifact.
   In the latter case, artifact name will be inferred from model configuration.
   
    An example with a local path:
    ```
    python retrieve.py ++ckpt_path=<local_path>
    ```

    An example with a W&B artifact:
    ```
    python retrieve.py +model=codet5 ++input.train_with_history=X ++input.encoder_input_type=X
    ```
3. Initialize RACE with fine-tuned CodeT5 weights and use retrieved examples to train the model. 
   Refer to the instruction above for details.
    
    > :construction: Currently, downloading retrieved predictions and fine-tuned CodeT5 checkpoint is only possible with W&B.

### Step 4: Evaluate

#### Step 4.1: Generating predictions

1. Define configuration for evaluation at [`conf/eval_config.py`](conf/eval_config.py).

2. Note that you have to either provide local path to checkpoint in `ckpt_path` or use W&B artifact.

   In the latter case, artifact name will be inferred from model configuration. Choose one of available model configs or add your own. 

3. Note that you have to define all parameters from [`InputConfig`](conf/data/input_config.py). You can do it via CLI or just rewrite them. Below is the example how to define parameters via CLI.

To launch evaluation of model defined as `XXXModelConfig` and registered via `ConfigStore.store(name="XXX", group="model", node=XXXModelConfig)`, run the following command:
```
python eval.py +model=XXX ++input.train_with_history=X ++input.encoder_input_type=X ++input.generate_with_history=X ++input.context_ratio=X
```

#### Step 4.2: Compute metrics

1. Define configuration for metrics computation at [`conf/metrics_config.py`](conf/metrics_config.py).

2. Note that you have to either provide local path to model predictions in `preds_path` or use W&B artifact and define the following parameters from [`ArtifactMetricsConfig`](conf/metrics_config.py): `name`, `version`. You can do it via CLI or just rewrite them. Below are the examples how to define parameters via CLI.


To launch metrics computation for local predictions:
```
python compute_metrics.py ++preds_path=XXX
```

To launch metrics computation for W&B artifact with predictions:
```
python compute_metrics.py ++logger.artifact_config.name=XXX ++logger.artifact_config.version=XXX
```
