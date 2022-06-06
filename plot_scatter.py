import numpy as np
import pandas as pd
import wandb

name = "transformer_with_history"
artifact_name = f"saridormi/cmg_eval/{name}_scores:v3"
table_name = "with_history_context_ratio_0.25"
wandb.Table.MAX_ROWS = 50000
with wandb.init(project="cmg_eval", name=name, tags=["scatter", name]) as run:
    table = run.use_artifact(artifact_name).get(table_name)
    df = pd.DataFrame(data=table.data, columns=table.columns)
    for diff_col in ["num_mods", "num_tokens_diff", "len_diff"]:
        df = df.loc[df[diff_col] <= np.percentile(df[diff_col], q=95)]
    table_cropped = wandb.Table(dataframe=df)
    wandb.log(
        {
            f"{metric_col}_{feature_col}": wandb.plot.scatter(
                table, feature_col, metric_col, title=f"{feature_col} vs. {metric_col}"
            )
            for metric_col in ["edit_similarity@1", "edit_similarity@2", "edit_similarity@5"]
            for feature_col in ["num_mods", "len_diff", "num_tokens_diff", "len_msg", "num_tokens_msg"]
        }
    )
    wandb.log(
        {
            f"{metric_col}_{feature_col}_cropped": wandb.plot.scatter(
                table_cropped, feature_col, metric_col, title=f"{feature_col} vs. {metric_col} (cropped)"
            )
            for metric_col in ["edit_similarity@1", "edit_similarity@2", "edit_similarity@5"]
            for feature_col in ["num_mods", "len_diff", "num_tokens_diff", "len_msg", "num_tokens_msg"]
        }
    )
