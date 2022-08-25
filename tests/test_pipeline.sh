# run seq2seq pipeline on first 100 examples
python train.py +model=random_roberta_6_random_gpt2_6 ++wandb_logger.project=test ++trainer.accelerator=gpu ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=random_roberta_6_random_gpt2_6 ++dataset.context_ratio=0.25 ++wandb_logger.project=test ++model_artifact.project=saridormi/test ++trainer.accelerator=gpu ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.model_name=random_roberta_6_random_gpt2_6_with-history ++wandb.model_config=context-ratio_0.25_with-history ++wandb.project=test ++wandb.artifact_project=saridormi/test
# run decoder-only pipeline on first 100 examples
python train.py +model=distilgpt2 ++wandb_logger.project=test ++trainer.accelerator=gpu ++trainer.devices=1 ++trainer.limit_train_batches=100 ++trainer.limit_val_batches=100
python eval.py +model=distilgpt2 ++dataset.context_ratio=0.25 ++wandb_logger.project=test ++model_artifact.project=saridormi/test ++trainer.accelerator=gpu ++trainer.devices=1 ++trainer.limit_test_batches=10
python compute_metrics.py ++wandb.model_name=distilgpt2_with-history ++wandb.model_config=context-ratio_0.25_with-history ++wandb.project=test ++wandb.artifact_project=saridormi/test