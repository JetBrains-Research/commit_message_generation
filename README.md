# Commit messages completion ~~(and generation)~~

![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge)

This repository contains code for training and evaluation of Transformer-based models for commit messages completion
task.

## Pretrained models and dataset

Several pretrained models are
available [as Weights & Biases artifacts](https://wandb.ai/saridormi/commit_message_generation/artifacts/model). You can
simply download them or do whatever can be done with artifacts *(
see [Weights & Biases docs](https://docs.wandb.ai/guides/artifacts) for more information)*.

Also, as part of this project, data about ~1.6M commits from open GitHub repositories was collected and preprocessed,
see [this repo](https://github.com/saridormi/commits_dataset) for more information.

## Usage

> **OS:** Linux

To use this project, follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commit_message_generation.git
    ```
2. **Install dependencies**

   This project has the following prerequisites:
    * Python 3.8
    * Neural networks frameworks: [PyTorch](https://pytorch.org/)
      , [🤗 Transformers](https://huggingface.co/transformers/)
      and [PyTorch Lightning](https://www.pytorchlightning.ai/)
    * Configuration: [Hydra](https://hydra.cc/)
    * Experiments tracking: [Weights & Biases](https://wandb.ai/site)
    * Metrics: [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
      , [🤗 Datasets](https://huggingface.co/docs/datasets/)
      and other packages necessary for specific metrics implementations
    * Lint & unit tests: [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/)
      , [pytest](https://docs.pytest.org/en/7.1.x/)

   You can install Python packages with [pip](https://pip.pypa.io/en/stable/):
    ```
    pip install -r requirements.txt
    ```
   Or with [conda](https://docs.conda.io/en/latest/):
    ```
    conda env create -f environment.yml
    ```
3. **Prepare data**

   You can use script from [this repo](https://github.com/saridormi/commits_dataset) for data preparation.

    <details>
    <summary>:yellow_heart: click here for more information on data format</summary>

   This projects expects input data to be already tokenized. Each dataset part (e.g. train, val, test) should be stored
   in two files: `part.json` and `part_history.json`.

    * `part.json`

      It is a JSON Lines file. Each row is a dictionary with the following keys: `diff_input_ids`, `pos_in_history`
      , `author`.

        * `diff_input_ids`: A tokenized representation of diff from current commit, basically, a list of tokens.
        * `pos_in_history`: An integer denoting what position current commit has in the commit history of its author.
        * `author`: An unique id for author of current commit.

    * `part_history.json`

      It is a JSON file. It contains a dictionary where each key is an unique author id and a corresponding value is the
      sequence of tokenized representation of commit messages from the author in chronological order.

  </details>

4. **Configure training and/or evaluation**

    * Configuration for training is defined at [`conf/train_config.yaml`](conf/train_config.yaml)

       <details>
       <summary>:yellow_heart: click here for more information on training config format</summary>

      Basically, config looks like that:

       ```
       dataset:
         kwarg: ...
       logger:
         kwarg: ...
       model:
         kwarg: ...
       trainer:
         kwarg: ...
       artifact:
         kwarg: ...
      ```

      See more information about possible options below.

        * `dataset` defines everything data-related

            * `dataset_root`: your path to dataset

            * `use_mtests`: **true** if you want to run additional validation on a small set of "marker tests" and **
              false**
              otherwise

            * `marker_tests_root`: your path to marker tests

            * `training_with_history`: **true** if you want to use previous message history during training and **
              false**
              otherwise

            * `encoder_context_max_len`: maximum allowed number of tokens in encoder context

            * `decoder_context_max_len`: maximum allowed number of tokens in decoder context

            * `encoder_name_or_path`: pretrained model name or path for **diff tokenizer** *(
              see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
              for additional info)*

            * `decoder_name_or_path`: pretrained model name or path for **message tokenizer** *(
              see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
              for additional info)*

            * `train_dataloader_conf` and etc. are passed to corresponding DataLoaders *(
              see [PyTorch docs](https://pytorch.org/docs/1.7.0/data.html#torch.utils.data.DataLoader) for additional
              info)*

        * `logger` defines everything logging-related

            * `_target_`: logger object that you want to use *(for Weights & Biases
              it's `pytorch_lightning.loggers.WandbLogger`,
              see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/logging.html#supported-loggers)
              for other options)*

            * everything else is passed to logger object as kwargs

        * `model` defines everything model-related

          Note that this project supports full encoder-decoder Transformer model and Transformer decoder model.

            * `encoder_decoder`:  **true** if you want to use full encoder-decoder Transformer and **false** if you want
              to use Transformer decoder

            1. Encoder-decoder configuration:

                * `learning_rate`: pretty self-explanatory, but note
                  that [`get_linear_schedule_with_warmup`](https://huggingface.co/transformers/v4.2.2/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup)
                  is used so this learning rate value is maximum and it is reached after 4000 steps
                * `decoder_name_or_path`: pretrained model name or path for **decoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*
                * `encoder_name_or_path`: pretrained model name or path for **encoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*
                * `num_layers_encoder`: number of layers in **encoder**
                * `num_layers_decoder`: number of layers in **decoder**

               You have to specify either `num_layers` for training from scratch or `name_or_path` for loading
               pretrained models. You can also specify `num_layers` for pretrained models, if it is less than actual
               number of layers in pretrained checkpoint, `num_layers` layers will be chosen uniformly.

            2. Decoder-only configuration:

                * `learning_rate`: pretty self-explanatory, but note
                  that [`get_linear_schedule_with_warmup`](https://huggingface.co/transformers/v4.2.2/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup)
                  is used so this learning rate value is maximum and it is reached after 4000 steps
                * `decoder_name_or_path`: pretrained model name or path for **decoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*

        * `trainer` defines everything trainer-related

          All options from here are passed to Trainer as kwargs *(
          see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/trainer.html) for additional
          info)*

        * `artifact` defines configuration for saving Weights & Biases artifact

          There is a possibility to save model checkpoints after training as Weights & Biases artifact. All options from
          here are passed to wandb.Artifact as kwargs *(see [W&B docs](https://docs.wandb.ai/ref/python/artifact) for
          additional info)*
      </details>

    * Configuration for generating predictions for the test set is defined
      at [`conf/eval_config.yaml`](conf/eval_config.yaml)

      <details>
      <summary>:yellow_heart: click here for more information on evaluation config format</summary>

      Basically, config looks like that:

      ```
      stage: test
      dataset:
        kwarg: ... 
      logger:
        kwarg: ...
      model:
        kwarg: ...
      trainer:
        kwarg: ...
      artifact:
        kwarg: ...
      ckpt_path: ...
      ```

      See more information about possible options below.

        * `stage` is set to `test` by default, it might be set to `sweep` to use validation set instead of test when
          tuning hyperparameters with W&B sweep

        * `dataset` defines everything data-related

            * `dataset_root`: your path to data

            * `generation_with_history`: **true** if you want to use previous message history during evaluation and **
              false** otherwise

            * `encoder_context_max_len`: maximum allowed number of tokens in encoder context

            * `decoder_context_max_len`: maximum allowed number of tokens in decoder context

            * `context_ratio`: a ratio of target message characters that go into context

            * `encoder_name_or_path`: pretrained model name or path for **diff tokenizer** *(
              see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
              for additional info)*

            * `decoder_name_or_path`: pretrained model name or path for **message tokenizer** *(
              see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
              for additional info)*

            * `test_dataloader_conf` is passed to corresponding DataLoader *(
              see [PyTorch docs](https://pytorch.org/docs/1.7.0/data.html#torch.utils.data.DataLoader) for additional
              info)*

        * `logger` defines everything logging-related

            * `_target_`: logger object that you want to use *(for Weights & Biases
              it's `pytorch_lightning.loggers.WandbLogger`,
              see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/logging.html#supported-loggers)
              for other options)*

            * everything else is passed to logger object as kwargs

        * `model` defines everything model-related

          Note that this project supports full encoder-decoder Transformer model and Transformer decoder model.

            * `encoder_decoder`:  **true** if you want to use full encoder-decoder Transformer and **false** if you want
              to use Transformer decoder

            1. Encoder-decoder configuration:

                * `decoder_name_or_path`: pretrained model name or path for **decoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*
                * `encoder_name_or_path`: pretrained model name or path for **encoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*
                * `num_layers_encoder`: number of layers in **encoder**
                * `num_layers_decoder`: number of layers in **decoder**

               You have to specify either `num_layers` for initializing model randomly or `name_or_path` for loading
               pretrained models. You can also specify `num_layers` for pretrained models, if it is less than actual
               number of layers in pretrained checkpoint, `num_layers` layers will be chosen uniformly.

            2. Decoder-only configuration:

                * `decoder_name_or_path`: pretrained model name or path for **decoder** *(
                  see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained)
                  for additional info)*

        * `trainer` defines everything trainer-related

          All options from here are passed to Trainer as kwargs *(
          see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/trainer.html) for additional
          info)*

        * `artifact`

          There is a possibility to use W&B artifact with model checkpoint for evaluation.
            * `name`: full W&B artifact name
            * `artifact_path`: a path to specific file to download within W&B artifact
            * `local_path`: a path where specified file will be saved locally

        * `ckpt_path`

          Another option: when you already have a model checkpoint saved locally, provide a path to checkpoint here.
      </details>

    * Configuration for computing metrics is defined at [`conf/metrics_config.yaml`](conf/metrics_config.yaml)

      <details>
      <summary>:yellow_heart: click here for more information on metrics config format</summary>

      Basically, config looks like that:

      ```
      wandb:
        kwarg: ...
      input_file: ...
      max_n_tokens: ...
      ```

      See more information about possible options below.

        * `wandb` defines everything Weights & Biases-related

          Firstly, there is an option to use model predictions stored as W&B artifact table. The following options
          define the configuration:
            * `artifact_name`: full W&B artifact name
            * `table_name`:  a name for specific table in the artifact

          Secondly, there is an option to log metrics to W&B in a separate run. The following options define the
          configuration:
            * `project`: W&B project name
            * `name`:  W&B run name

        * `input_file`

          An alternative to W&B: if you have model predictions stored as `.csv` file locally, provide the path here.

        * `max_n_tokens`

          Metrics are computed both for full predictions and references and for their prefixes of first `i` tokens,
          where `i` goes from `1` to `max_n_tokens + 1`.
      </details>

5. **Train**

   To train a model, define configuration at [`conf/train_config.yaml`](conf/train_config.yaml) and run the following
   command:
    ```
    python train.py
    ```

6. **Evaluate**

   To generate predictions for the test set, define configuration at [`conf/eval_config.yaml`](conf/eval_config.yaml)
   and run the following command:

    ```
    python eval.py
    ```

   To compute metrics, define configuration at [`conf/metrics_config.yaml`](conf/metrics_config.yaml) and run the
   following command:

    ```
    python compute_metrics.py
    ```
