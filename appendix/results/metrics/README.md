# Metrics

In this directory, we provide metric values for all our experiments.

* [`cmg_approaches`](cmg_approaches) – results for CMG approaches on $CMG_{test}$.
* [`llm`](llm) – results for CMG approaches and LLM GPT-3.5-turbo on $LLM_{test}$.
* [`filters`](filters) – results for CMG approaches on Filtered, Out-of-Filters and Random subsets of $CMG_{test}$ with 10,385 examples.

For each setting, we provide:
* `full_metrics.jsonl` – metrics between full predictions and targets, stored in JSONLines format.
* `prefix_metrics.jsonl` – metrics between prefixes of predictions and targets, stored in JSONLines format.
