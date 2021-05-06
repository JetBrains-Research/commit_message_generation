# Commit messages completion ~~(and generation)~~
![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge) 

This repository contains code for training and evaluation of Transformer-based models for commit messages completion task.

## Pretrained models and dataset

Several pretrained models are available [as Weights & Biases artifacts](https://wandb.ai/saridormi/commit_message_generation/artifacts/model). You can simply download them or do whatever can be done with artifacts *(see [Weights & Biases docs](https://docs.wandb.ai/guides/artifacts) for more information)*.

Also, as part of this project, data about ~1.3M commits from open GitHub repositories was collected and preprocessed *(later I'll add more information)*.  Dataset is available [at Google Drive](https://drive.google.com/drive/folders/1MgwVpvD2QYL5F3xZGdmb40T7RHdQEe7y)  or [as Weights & Biases artifact](https://wandb.ai/saridormi/commit_message_generation/artifacts/dataset) as well. 

## Usage
> **OS:** Linux (should work on Windows too, but I haven't fully tested it)

To use this project, follow these steps:

1. **Clone repo**
    ```
    git clone https://github.com/saridormi/commit_message_generation.git
    ```
2. **Install dependencies**

    This project has the following prerequisites:
    * Python 3.8
    * Neural networks frameworks: [PyTorch](https://pytorch.org/), [🤗 Transformers](https://huggingface.co/transformers/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)
    * Configuration: [Hydra](https://hydra.cc/)
    * Experiments tracking: [Weights & Biases](https://wandb.ai/site)
    * Metrics-related stuff: [🤗 Datasets](https://huggingface.co/docs/datasets/), [NLTK](https://www.nltk.org/), [rouge-score](https://pypi.org/project/rouge-score/)

    You can install Python packages with [pip](https://pip.pypa.io/en/stable/):
    ```
    pip install -r requirements.txt
    ```
    Or with [conda](https://docs.conda.io/en/latest/):
    ```
    conda env create -f environment.yml
    ```
3. **Prepare data**
- Using existing data

You can download tokenized and ready-to-use dataset [here](https://drive.google.com/drive/folders/1MgwVpvD2QYL5F3xZGdmb40T7RHdQEe7y) (`tokenized` folder).

- Using your own data

> ❗ When project runs for the first time, data gets preprocessed and tokenized, which can take a long time.
    
Current version assumes that data is stored in `.csv` files with three columns: `diff`, `message` and `author`.

<details>
<summary>:yellow_heart: More information on data format</summary>

* Diff is basically `git diff` output string but some special info like `index e345a66..f841d45` or `@@ -6,22 +6,24 @@` is omitted and it additionally contains special token `<FILE>` in lines with filenames. 
* Message is, well, commit message. 

 Note that in both cases input lines are separated with `<nl>` token and punctuation is additionally separated by whitespaces.
* Author can be anything that can be used as a dictionary key, e.g. some `id` integer or `name` string or `(name, email)` tuple.

Super simple examples of data format in cases of modifying, adding, deleting or renaming file:
|author|diff|message|
|:-:|:-:|:-:|
|1|<FILE> conf / config . yaml <nl> - batch_size : 4 <nl> + batch_size : 8|Change config|
|2|new file <nl> <FILE> conf / config . yaml <nl> + batch_size : 8|Add config|
|1|deleted file <nl> <FILE> conf / config . yaml <nl> - batch_size : 4|Remove config|
|2|rename from conf / config . yaml <nl> rename to conf / conf . yaml|Rename config|
</details>

4. **Configure training and/or evaluation**

Configuration is defined at `conf/config.yaml`. Basically, it looks like that:

```
dataset:
  arg: ...
logger:
  arg: ...
model:
  arg: ...
trainer:
  arg: ...
ckpt_path: ...
```

See more information about possible options below.

<details>
<summary>:yellow_heart: dataset</summary>

Defines everything data-related

* `dataset_root`: your path to data
        
* `with_history`: **true** if you want to use previous message history during training and evaluation and **false** otherwise
        
* `history_max_len`: maximum allowed number of tokens in previous message history and message combined
        
* `encoder_name_or_path`: pretrained model name or path for **diff tokenizer** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
        
* `decoder_name_or_path`: pretrained model name or path for **message tokenizer** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
        
* `local_rank` and `world_size` are needed for multi-GPU setup, they are set automatically
        
* `train_dataloader_conf` and etc. are passed to corresponding dataloaders *(see [PyTorch docs](https://pytorch.org/docs/1.7.0/data.html#torch.utils.data.DataLoader) for additional info)*
</details>

<details>
<summary>:yellow_heart: logger</summary>
Defines everything logging-related

* `_target_`: logger object that you want to use *(for Weights & Biases it's `pytorch_lightning.loggers.WandbLogger`, see [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/logging.html#supported-loggers) for other options)*

* everything else is passed to logger object as kwargs
</details>

<details>
<summary>:yellow_heart: model</summary>
Defines everything model-related

Note that this project supports full encoder-decoder Transformer model and Transformer decoder model.

* `encoder_decoder`:  **true** if you want to use full encoder-decoder Transformer and **false** if you want to use Transformer decoder

Let's discuss these two cases separately.

1. Encoder-decoder:

  * `learning_rate`: pretty self-explanatory, but note that [`get_linear_schedule_with_warmup`](https://huggingface.co/transformers/v4.2.2/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup) is used so this learning rate value is maximum and it is reached after 4000 steps
  * `decoder_name_or_path`: pretrained model name or path for **decoder** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
  * `encoder_name_or_path`: pretrained model name or path for **encoder** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
  * `num_layers_encoder`: number of layers in **encoder**
  * `num_layers_decoder`: number of layers in **decoder**

You have to specify either `num_layers` for training from scratch or `name_or_path` for loading pretrained models. You can also specify `num_layers` for pretrained models, if it is less than actual number of layers in pretrained checkpoint, `num_layers` layers will be chosen uniformly.

2. Decoder-only:
  
  * `learning_rate`: pretty self-explanatory, but note that [`get_linear_schedule_with_warmup`](https://huggingface.co/transformers/v4.2.2/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup) is used so this learning rate value is maximum and it is reached after 4000 steps
  * `decoder_name_or_path`: pretrained model name or path for **decoder** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
</details>

<details>
<summary>:yellow_heart: trainer</summary>
Defines everything trainer-related

All options from here are passed to Trainer as kwargs. See [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/1.1.4/trainer.html) for more information.
</details>

<details>
<summary>:yellow_heart: ckpt_path</summary>
Provide a path to pretrained model checkpoint here to evaluate it.
</details>

5. **(optional) Train**
    
    To train a model, define configuration at `conf/config.yaml` and run the following command:
    ```
    python train.py
    ```

    Note that evaluation is done at the end too, next step is for cases when you only want to evaluate some pretrained model.
    
6. **Evaluate**

    To evaluate a model, define configuration at `conf/config.yaml` and run the following command:
    ```
    python eval.py
    ```
