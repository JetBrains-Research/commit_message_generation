# From Commit Message Generation to History-Aware Commit Message Completion

![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge)

<p align="center">
| <a href="TODO">:scroll: Preprint</a> | <a href="https://huggingface.co/datasets/JetBrains-Research/commit-chronicle"> :hugs: Dataset</a> | <a href="https://huggingface.co/JetBrains-Research/cmg-codet5-without-history#available-checkpoints">:hugs: Models</a> |
</p>

This repository provides a replication package for our paper :scroll: From Commit Message Generation to History-Aware Commit Message Completion, ASE 2023.

* **Code**
  * Models experiments – this repository
    * The most actual version – `main` branch
    * The exact replication package for CMG experiments for our ASE 2023 paper – [`appendix_cmg` tag](https://github.com/JetBrains-Research/commit_message_generation/tree/appendix_cmg)
    * The exact replication package for LLM experiments for our ASE 2023 paper – [`appendix_llm` tag](https://github.com/JetBrains-Research/commit_message_generation/tree/appendix_llm)
  * Data collection and processing – [separate repo](https://github.com/saridormi/commits_dataset)
* **Dataset**: also available on [Zenodo](https://zenodo.org/record/8189044)
* **Models checkpoints**: also available on [Zenodo](https://zenodo.org/record/8199408)
* **Other**
  * Models predictions: [`appendix/predictions`](appendix/predictions)
  * Full experimental results: [`appendix/results`](appendix/results)

> :construction: Work in progress, some links are currently unavailable.

## How to use

### Requirements

* :snake: Python
* :floppy_disk: Dependencies
    * This project provides dependencies for two Python dependency managers:
      * Poetry: [`poetry.lock`](poetry.lock), [`pyproject.toml`](pyproject.toml) (preferred)
      * pip: [`requirements.txt`](requirements.txt) (obtained through `poetry export --with dev,retrieval --output requirements.txt`)

### Usage

#### Step 1: Prepare raw data

> :star2: Useful links: [our dataset](https://huggingface.co/datasets/JetBrains-Research/commit-chronicle) and/or [the repo](https://github.com/saridormi/commits_dataset) we used for data preparation. 

<details>
<summary>:yellow_heart: click here for more information on required data format</summary>

This project expects each dataset part to be stored in a separate JSONLines files:
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

#### Step 2: Choose configuration

##### Model architecture

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

##### Input type

This project explores two kinds of input for a commit message completion task: diff and commit message history. 

* For decoder, there is only one supported option: concatenate commit message history with a current commit message and pass to context.

* For seq2seq models, there are three supported options:
  * *Diff-only:* pass diff to encoder, pass a current message to decoder.
  * *History-only:* pass history to encoder, pass a current message to decoder.
  * *Diff + history:* pass diff to encoder, pass commit message history concatenated with a current message to decoder.

#### Step 3: Train

1. Define configuration for training at [`conf/train_config.py`](conf/train_config.py).
2. Choose one of available model configs or add your own.
3. Note that you have to define missing parameters from [`InputConfig`](conf/data/input_config.py). You can do it via CLI or just rewrite them. Below is the example how to define parameters via CLI.

To launch training of model defined as `XXXModelConfig` and registered via `ConfigStore.store(name="XXX", group="model", node=XXXModelConfig)`, run the following command (with actual values instead of X's):
```
python train.py +model=XXX ++input.train_with_history=X ++input.encoder_input_type=X
```

##### Additional steps for RACE model

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

   For checkpoint, you have to either provide a path to checkpoint in :hugs: Transformers format as `name_or_path` in [`RACEConfig`](conf/model/configs.py) or
   define [`logger.checkpoint`](conf/train_config.py) in train config correctly to download it from W&B Artifacts.
   
   For retrieved examples, you have to either provide them locally or define [`logger.retrieval`](conf/train_config.py) in train config correctly to download it from W&B Artifacts.
   
   To provide retrieved examples locally, place them inside root dataset directory in a folder `retrieval_with_history` or `retrieval_without_history` (depending whether the encoder used for retrieval was trained with history or not).

    ```
     ├── ...  # data directory
     │   ├── retrieval_with_history
     │   │    ├── train_predictions.jsonl
     │   │    ├── val_predictions.jsonl
     │   │    ├── test_predictions.jsonl
     │   ├── retrieval_without_history
     │   │    ├── train_predictions.jsonl
     │   │    ├── val_predictions.jsonl
     │   │    ├── test_predictions.jsonl
     │   ├── train.jsonl
     │   ├── val.jsonl
     │   └── test.jsonl
     └── ...
    ```
#### Step 4: Evaluate

##### Step 4.1: Generating predictions

1. Define configuration for evaluation at [`conf/eval_config.py`](conf/eval_config.py).

2. Note that you have to either provide local path to checkpoint in `ckpt_path` or use W&B artifact.

   In the latter case, artifact name will be inferred from model configuration. Choose one of available model configs or add your own. 

3. Note that you have to define all parameters from [`InputConfig`](conf/data/input_config.py). You can do it via CLI or just rewrite them. Below is the example how to define parameters via CLI.

To launch evaluation of a model defined as `XXXModelConfig` and registered via `ConfigStore.store(name="XXX", group="model", node=XXXModelConfig)`, run the following command:
```
python eval.py +model=XXX ++input.train_with_history=X ++input.encoder_input_type=X ++input.generate_with_history=X ++input.context_ratio=X
```

##### Step 4.2: Compute metrics

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
