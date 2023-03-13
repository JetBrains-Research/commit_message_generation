# a simple script that launches pipeline for all kinds of models
# run to make sure that artifacts & everything else in W&B project is setup as intended
echo "W&B username: $1"
echo "Accelerator: $2 (should be one of 'gpu','cpu')"

# run codet5 pipeline with diffs & history on first 10 examples
python train.py +model=codet5 ++input.train_with_history=true ++logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=10 ++trainer.limit_val_batches=10 ++dataset.use_cache=true ++dataset.train_dataloader_conf.batch_size=4 ++dataset.val_dataloader_conf.batch_size=4 ++input.encoder_input_type=diff
python eval.py +model=codet5 ++input.train_with_history=true ++input.context_ratio=0.25 ++input.generate_with_history=true ++logger.project=test ++logger.artifact_config.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10 ++dataset.use_cache=true ++input.encoder_input_type=diff
python compute_metrics.py ++logger.project=test ++logger.artifact_config.name=codet5_with-history_preds ++logger.artifact_config.project="$1/test" ++logger.artifact_config.version=context-ratio_0.25_with-history

# run codereviewer pipeline with diffs & history on first 10 examples
python train.py +model=codereviewer ++input.train_with_history=true ++logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=10 ++trainer.limit_val_batches=10 ++dataset.use_cache=true ++dataset.train_dataloader_conf.batch_size=4 ++dataset.val_dataloader_conf.batch_size=4 ++input.encoder_input_type=diff
python eval.py +model=codereviewer ++input.train_with_history=true ++input.context_ratio=0.25 ++input.generate_with_history=true ++logger.project=test ++logger.artifact_config.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10 ++dataset.use_cache=true ++input.encoder_input_type=diff
python compute_metrics.py ++logger.project=test ++logger.artifact_config.name=codereviewer_with-history_preds ++logger.artifact_config.project="$1/test" ++logger.artifact_config.version=context-ratio_0.25_with-history

# run race pipeline with diffs & history on first 10 examples
python train.py +model=race ++input.train_with_history=true ++logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=10 ++trainer.limit_val_batches=10 ++dataset.use_cache=false ++dataset.train_dataloader_conf.batch_size=4 ++dataset.val_dataloader_conf.batch_size=4 ++input.encoder_input_type=diff
python eval.py +model=race ++input.train_with_history=true ++input.context_ratio=0.25 ++input.generate_with_history=false ++logger.project=test ++logger.artifact_config.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10 ++dataset.use_cache=true ++input.encoder_input_type=diff
python compute_metrics.py ++logger.project=test ++logger.artifact_config.name=race_with-history_preds ++logger.artifact_config.project="$1/test" ++logger.artifact_config.version=context-ratio_0.25_with-history

# run distilgpt2 pipeline on first 10 examples
python train.py +model=distilgpt2 ++input.train_with_history=true ++input.encoder_input_type=diff ++logger.project=test ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_train_batches=10 ++trainer.limit_val_batches=10 ++dataset.use_cache=false ++dataset.train_dataloader_conf.batch_size=4 ++dataset.val_dataloader_conf.batch_size=4 ++input.encoder_input_type=diff
python eval.py +model=distilgpt2 ++input.train_with_history=true ++input.encoder_input_type=diff ++input.generate_with_history=true ++input.context_ratio=0.25 ++logger.project=test ++logger.artifact_config.project="$1/test" ++trainer.accelerator="$2" ++trainer.devices=1 ++trainer.limit_test_batches=10 ++dataset.use_cache=true ++input.encoder_input_type=diff
python compute_metrics.py ++logger.project=test ++logger.artifact_config.name=distilgpt2_with-history_preds ++logger.artifact_config.project="$1/test" ++logger.artifact_config.version=context-ratio_0.25