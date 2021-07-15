# Generation pipeline
> :exclamation: work in progress 

This branch contains code for performing all possibly necessary steps for commit messages completion.

## Usage

1. **Install dependencies**

    This project has the following prerequisites:
    * Python 3.8
    * Code quality: [Black](https://github.com/psf/black), [Mypy](https://github.com/python/mypy) and [pytest](https://github.com/pytest-dev/pytest/)
    * Neural networks frameworks: [PyTorch](https://pytorch.org/), [🤗 Transformers](https://huggingface.co/transformers/)
    * Configuration: [Hydra](https://hydra.cc/)
    
    You can install Python packages with [pip](https://pip.pypa.io/en/stable/):
    ```
    pip install -r requirements.txt
    ```

3. **Define configuration**

Write `.yaml` config according to your needs and put it into `configs` folder. 

<details>
<summary>:yellow_heart: click to see more information</summary>

Some default config example:

```
data_processor:
  prompt_max_len: 200
  diff_tokenizer_name_or_path: distilbert-base-cased
  msg_tokenizer_name_or_path: distilgpt2
  preprocessing: false
model:
  encoder_name_or_path: distilbert-base-cased
  decoder_name_or_path: distilgpt2
generation_kwargs:
  num_beams: 5
  num_return_sequences: 5
device: cpu
min_length: 5
max_length: 5
```

Possible options:

* `data_processor`

    > Defines everything related to data processing

    * `prompt_max_len`: maximum allowed number of tokens in previous message history and current message combined

    * `diff_tokenizer_name_or_path`: pretrained name or path for **diff tokenizer** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*

    * `msg_tokenizer_name_or_path`: pretrained name or path for **message tokenizer** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*

    * `preprocessing`: **true** if you want to preprocess data (remove unchanged lines from diffs and etc.), **false** 
      otherwise *(**false** by default)*

    * `nl_token`: newline character in your data *(`\n` by default)*

* `model`

    > Defines everything related to model

    * `decoder_name_or_path`: pretrained model name or path for **decoder** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*
    * `encoder_name_or_path`: pretrained model name or path for **encoder** *(see [HuggingFace docs](https://huggingface.co/transformers/v4.2.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained) for additional info)*

* `generation_kwargs`

    All kwargs from here are passed to `generate` method of `GPT2Decoder`, 
    which has almost the same signature as in `generate` method from `transformers`, so
    see [`transformers` docs](https://huggingface.co/transformers/v4.2.2/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate) for more information.

* `device`: `cuda:smth` to run pipeline on GPU or `cpu` to run on CPU
 
* `min_length`: minimum allowed number of tokens to generate (integer)
 
* `max_length`: maximum allowed number of tokens to generate (integer)
</details>

3. **Use `generate` function**

> :question: For now this function is assumed to be called from elsewhere, 
> so you have to provide as parameters already initialized model and data processor and all
> inputs for generation

This function covers all possibly necessary steps:
1. *(optional)* Preprocess inputs
2. Tokenize inputs, concatenate history with message for generation context
3. *(optional)* Run encoder on diff
4. Run beam search generation conditioned on generation context and encoder outputs
