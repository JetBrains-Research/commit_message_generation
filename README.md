# Commit messages completion ~~(and generation)~~

![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge)

This repository contains code for training and evaluation of Transformer-based models for commit messages completion
task.

## Models checkpoints and dataset

> TODO: later we'll share information about available models checkpoints and dataset

## Usage

> **OS:** Linux

To use this project, follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commit_message_generation.git
    ```
2. **Install dependencies**

   This project has the following prerequisites:
    * Python
    * Neural networks frameworks: [PyTorch](https://pytorch.org/)
      , [🤗 Transformers](https://huggingface.co/transformers/)
      and [PyTorch Lightning](https://www.pytorchlightning.ai/)
    * Configuration: [Hydra](https://hydra.cc/)
    * Experiments tracking: [Weights & Biases](https://wandb.ai/site)
    * Metrics: [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
      , [🤗 Datasets](https://huggingface.co/docs/datasets/)
      and other packages necessary for specific metrics implementations
    * Lint & unit tests: [mypy](https://github.com/python/mypy), [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/)
      , [pytest](https://docs.pytest.org/en/7.1.x/)

   You can install Python packages with [pip](https://pip.pypa.io/en/stable/):
    ```
    pip install -r requirements.txt
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
       model:
         kwarg: ...
       wandb_logger:
         kwarg: ...
       trainer:
         kwarg: ...
      ```

      See more information about possible options below.

        * `dataset` defines everything data-related. 
        
            You can check [corresponding DataModule class](src/data_utils/cmc_data_module.py) for full list of parameters.
            
            Part of data configuration is related to specific model (e.g. tokenizer paths) and it is defined in model config!
            All other options are defined in separate config, e.g. [`conf/dataset/default_dataset.yaml`](conf/dataset/default_dataset.yaml):

            * `testing`: True to generate noise of maximum allowed context length instead of using real data, False otherwise (used for bach size-tuning purposes).
            * `context_ratio`: Relevant for generation: ratio of characters of target message that would be passed to model context. Float, should be in (0,1) range.
            * `train_with_history`: True if you want to use commit message history during training, False otherwise.
            * `generate_with_history`: True if you want to use commit message history during generation, False otherwise.
            * `encoder_input_type`: What type of input will be passed to encoder. Currently, `history` and `diff` are supported.
            * `train_dataloader_conf` and etc. are passed to corresponding DataLoaders *(
              see [PyTorch docs](https://pytorch.org/docs/1.7.0/data.html#torch.utils.data.DataLoader) for additional
              information)*.

        * `model` defines everything model-related.

            Only optimizer-related model parameters are defined at [`conf/train_config.yaml`](conf/train_config.yaml):
            * `learning_rate`: well, learning rate! But note that: 
                * [`get_linear_schedule_with_warmup`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) is used, so this learning rate value is maximum that it is reached after `num_warmup_steps` steps
                * this learning rate value will be multiplied by batch size
            * `weight_decay`: Float, will be passed to AdamW optimizer.
            * `num_warmup_steps`: Int, number of warmup steps for scheduler.
            
            Other parameters are defined in a separate model config. Currently, we have two supported model configurations: encoder-decoder Transformer and decoder-only Transformer.
            
            * There is an example available for **decoder-only Transformer**: [`distilgpt2`](conf/model/distilgpt2.yaml). 
              Check out [corresponding class](src/model/configurations/decoder_wrapper.py) for information on parameters.
          
            * There are two examples available for **encoder-decoder Transformer**: [`random_roberta_2_random_gpt2_2`](conf/model/random_roberta_2_random_gpt2_2.yaml) and [`distilgpt2_distilgpt2_shared`](conf/model/distilgpt2_distilgpt2_shared.yaml).
              Check out [corresponding class](src/model/configurations/encoder_decoder_wrapper.py) for information on parameters.

        * `wandb_logger` defines everything logging-related.
            
            You can set `wandb_logger` to False to avoid using W&B, then Tensorboard (default option in Lightning) will be used.

            * `project`: Name of W&B project.
            * `save_model_as_artifact`: True to upload model checkpoint to W&B after training.

        * `trainer` defines everything trainer-related.

          All options from here are passed to Trainer as kwargs *(see [Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) for additional information)*.
      </details>

    * Configuration for generating predictions for the test set is defined
      at [`conf/eval_config.yaml`](conf/eval_config.yaml)

      <details>
      <summary>:yellow_heart: click here for more information on evaluation config format</summary>

      Basically, config looks like that:

      ```
      stage: ...
      
      dataset:
        kwarg: ... 
      
      wandb_logger:
        kwarg: ...
      
      generation_kwargs:
        kwargs: ...
      
      trainer:
        kwarg: ...
      
      model:
        kwarg: ...
        
      model_name: ...
      
      artifact:
        kwarg: ...

      ckpt_path: ...
      ```

      See more information about possible options below.

        * `stage` is set to `test` by default, it might be set to `sweep` to use validation set instead of test when
          tuning hyperparameters with W&B sweep.

        * `dataset` defines everything data-related. 
           
          It is the same as in the train config, check the information above. 

        * `wandb_logger` defines everything logging-related.

            You can set `wandb_logger` to False to avoid using W&B, then Tensorboard (default option in Lightning) will be used.

            * `project`: Name of W&B project.
        
        * `generation_kwargs` defines parameters for generation.

            Check [HuggingFace's `generate` documentation](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) for more information

        Next, you can either:
        * use fine-tuned model
          * load from W&B artifact
            * define artifact parameters under `model_artifact` key
            * define `model_name` key (or define model configuration under `model` key, and `model_name` will be constructed automatically)
          * load from local path
           * provide local path under `ckpt_path` key
           * define `model_name` key (or define model configuration under `model` key, and `model_name` will be constructed automatically)
        * initialize random/pretrained model
          * define model configuration under `model` key (it is the same as in the train config, check the information above)
      </details>

    * Configuration for computing metrics is defined at [`conf/metrics_config.yaml`](conf/metrics_config.yaml)

      <details>
      <summary>:yellow_heart: click here for more information on metrics config format</summary>

      Basically, config looks like that:

      ```
      wandb:
        kwarg: ...
        artifact: 
          kwargs: ...
      
      input_file: ...
      max_n_tokens: ...
      
      language: ...
      only_short_sequences: ...
      only_long_sequences: ...
      ```

      See more information about possible options below.

        * `wandb` defines everything W&B-related

          Firstly, there is an option to use model predictions stored as W&B artifact table. Define the configuration under `artifact` key:
            * `project`: W&B project.
            * `name`:  Artifact name.
            * `version`: Artifact version (or alias).
            * `table_name`: Name of file with prediction in the artifact (by default, it is assumed to be the same as artifact alias).

          Secondly, there is an option to log metrics to W&B. The following options define the
          configuration:
            * `project`: W&B project name

        * `input_file`

          An alternative to W&B: if you have model predictions stored as `.csv` file locally, provide the path here.

        * `max_n_tokens`

          Metrics are computed both for full predictions and references and for their prefixes of first `i` tokens,
          where `i` goes from `1` to `max_n_tokens + 1`.
      
        Next, we also support a couple of filters:
        * `language`: Set to False to evaluate on full test set, set to programming language name to evaluate only on examples on this language.
        * `only_short_sequences`: Set to False to evaluate on full test set, set to True to evaluate only on examples with <= 512 tokens in diffs.
        * `only_long_sequences`: Set to False to evaluate on full test set, set to True to evaluate only on examples with > 512 tokens in diffs.
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
