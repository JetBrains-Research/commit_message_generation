# Supplementary Materials

> :bulb: Note: each folder has its own README with further details!

This folder contains:

* [`results`](results): comprehensive metrics for all our experiments, stored as JSONLines files;
* [`predictions`](predictions): model predictions for all our experiments, stored as JSONLines files;
* [`filters`](filters): the implementations of frequent filters from CMG research we used in our experiments.

### Other details referenced in the paper
* The prompts for LLMs are available under [`appendix_llm` tag](https://github.com/JetBrains-Research/commit_message_generation/tree/appendix_llm) in [`src/data_utils/cmg_prompts.py`](https://github.com/JetBrains-Research/commit_message_generation/blob/appendix_llm/src/data_utils/cmg_prompts.py).
* The regular expressions we used for commit messages processing are available in [another repo](https://github.com/saridormi/commit_chronicle) in [`src/processing/message_processor.py`](https://github.com/saridormi/commit_chronicle/blob/appendix/src/processing/message_processor.py).
* Specific percentiles that we used to drop outliers from our dataset are available in [`dataset_percentiles.md`](dataset_percentiles.md).
