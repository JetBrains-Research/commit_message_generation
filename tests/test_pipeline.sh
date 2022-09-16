# a simple script that launches all kinds of models in the pipeline
# run to make sure that artifacts & everything else in W&B project is setup as intended
echo "W&B username: $1"
echo "Accelerator: $2 (should be one of 'gpu','cpu')"

# run seq2seq pipeline with diffs first 100 examples
python train.py +model=random_roberta_2_random_gpt2_2 ++dataset.train_with_history=false ++wandb_logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=random_roberta_2_random_gpt2_2 ++dataset.train_with_history=false ++dataset.context_ratio=0.25 ++dataset.generate_with_history=false ++wandb_logger.project=test ++model_artifact.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.project=test ++wandb.artifact.name=random_roberta_2_random_gpt2_2_without-history_preds ++wandb.artifact.project="$1/test" ++wandb.artifact.version=context-ratio_0.25_without-history
# run seq2seq pipeline with history on first 100 examples
python train.py +model=random_roberta_2_random_gpt2_2 ++dataset.encoder_input_type=history ++wandb_logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=random_roberta_2_random_gpt2_2 ++dataset.encoder_input_type=history ++dataset.context_ratio=0.25 ++wandb_logger.project=test ++model_artifact.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.project=test ++wandb.artifact.name=random_roberta_2_random_gpt2_2_history-input_preds ++wandb.artifact.project="$1/test" ++wandb.artifact.version=context-ratio_0.25
# run seq2seq pipeline with diffs & history on first 100 examples
python train.py +model=random_roberta_2_random_gpt2_2 ++dataset.train_with_history=true ++wandb_logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=random_roberta_2_random_gpt2_2 ++dataset.generate_with_history=true ++dataset.context_ratio=0.25 ++wandb_logger.project=test ++model_artifact.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.project=test ++wandb.artifact.name=random_roberta_2_random_gpt2_2_with-history_preds ++wandb.artifact.project="$1/test" ++wandb.artifact.version=context-ratio_0.25_with-history
# run decoder-only pipeline on first 100 examples
python train.py +model=distilgpt2 ++wandb_logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=distilgpt2 ++dataset.context_ratio=0.25 ++wandb_logger.project=test ++model_artifact.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.project=test ++wandb.artifact.name=distilgpt2_with-history_preds ++wandb.artifact.project="$1/test" ++wandb.artifact.version=context-ratio_0.25