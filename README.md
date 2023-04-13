# Commit message completion ~~(and generation)~~



![GitHub](https://img.shields.io/github/license/saridormi/commit_message_generation?style=for-the-badge)

This branch provides code for experiments with OpenAI models for commit message generation task.

## Requirements

* :snake: Python

This project provides dependencies for two Python dependency managers:
* Poetry: [`poetry.lock`](poetry.lock), [`pyproject.toml`](pyproject.toml)
* pip: [`requirements.txt`](requirements.txt) (obtained through `poetry export --with dev --output requirements.txt`)
     
## Usage

### Step 1: Prepare raw data

> :construction: At some point, we plan to publish our dataset of commits. Until then, and if you wish to utilize this project 
> for other data, refer to this section.

> :star2: The data for this project was obtained via [commits_dataset](https://github.com/saridormi/commits_dataset) repo. 

<details>
<summary>:yellow_heart: click here for more information on required data format</summary>

This project expects input to be stored in a JSONLines format:
```
 ├── ...  # data directory
 │   ├── <input_file>.jsonl
 └── ...
```

In our case, each input example is commit. Specifically, the following keys are expected in each row:

* `message`: Commit message.
* `mods`: A list of modification made in a commit. Each modification should contain the following keys:
  * `change_type`: Type of modification (string, one of `MODIFY`, `ADD`, `DELETE`, `RENAME`, `COPY`, `UNKNOWN`).
  * `old_path`: Path to file before the commit (`None` when `change_type` is `ADD`).
  * `new_path`: Path to file after the commit (`None` when `change_type` is `DELETE`).
  * `diff`: Output of the `git diff` command for this specific file.

</details>

### Step 2: Generate predictions

Define configuration for evaluation at [`conf/openai_config.py`](conf/openai_config.py). 

Note that you have to define all the `MISSING` parameters. You can do it via CLI or just rewrite them. Below are the examples how to define parameters via CLI.

To launch evaluation, run the following command:
```
python eval_openai.py ++model_id=XXX ++dataset.prompt_configuration=XXX
```

### Step 3: Compute metrics

Define configuration for metrics computation at [`conf/metrics_config.py`](conf/metrics_config.py).

Note that you have to either provide local path to model predictions in `preds_path` or use W&B artifact and define the following parameters from [`ArtifactMetricsConfig`](conf/metrics_config.py): `name`, `artifact_path`. You can do it via CLI or just rewrite them. Below are the examples how to define parameters via CLI.


To launch metrics computation for local predictions:
```
python compute_metrics.py ++preds_path=XXX
```

To launch metrics computation for W&B artifact with predictions:
```
python compute_metrics.py ++logger.artifact_config.name=XXX ++logger.artifact_config.artifact_path=XXX
```
