import json
import os
import re
import subprocess


def prepare_data(tmp_path):
    for i, part in enumerate(["train", "val", "test"]):
        with open(f"{tmp_path}/{part}.jsonl", "w") as f:
            json.dump(
                {
                    "author": i,
                    "repo": f"sample_{part}_repo",
                    "hash": "sample hash",
                    "mods": [
                        {"change_type": "MODIFY", "old_path": "fname", "new_path": "fname", "diff": "sample diff"}
                    ],
                    "message": "sample commit message",
                    "license": "MIT License",
                    "language": "Python",
                },
                f,
            )
    return tmp_path


def test_train_pipeline(tmp_path):
    root_dir = prepare_data(tmp_path)

    if "train.py" not in os.listdir():
        os.chdir("..")

    for use_history in ["true", "false"]:
        command = (
            "python train.py +model=codet5 "
            "++input.encoder_input_type=diff "
            f"++input.train_with_history={use_history} "
            "++trainer.accelerator=cpu "
            "++trainer.devices=1 "
            "++trainer.max_epochs=1 "
            "++dataset.use_cache=false "
            "++dataset.use_eval_downsample=false "
            f'++dataset.dataset_root="{root_dir}" '
            "++dataset.train_dataloader_conf.batch_size=1 "
            "++dataset.val_dataloader_conf.batch_size=1 "
            "++logger.use_wandb=false "
            "++optimizer.learning_rate=0.00002 ++optimizer.weight_decay=0.0 ++optimizer.num_warmup_steps=100"
        )
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output_lines = re.split(r"[\n\r]", stdout.decode("utf-8"))
        assert any(line.startswith("Epoch 0: 100%") for line in output_lines)


def test_eval_pipeline(tmp_path):
    root_dir = prepare_data(tmp_path)

    if "eval.py" not in os.listdir():
        os.chdir("..")

    for use_history in ["true", "false"]:
        for context_ratio in [0.0, 0.5]:
            command = (
                "python eval.py +model=codet5 "
                "++input.encoder_input_type=diff "
                f"++input.train_with_history={use_history} "
                f"++input.generate_with_history={use_history} "
                f"++input.context_ratio={context_ratio} "
                "++trainer.accelerator=cpu "
                "++trainer.devices=1 "
                "++dataset.use_eval_downsample=false "
                "++dataset.use_cache=false "
                f'++dataset.dataset_root="{root_dir}" '
                "++dataset.test_dataloader_conf.batch_size=1 "
                "++logger.use_wandb=false"
            )
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output_lines = re.split(r"[\n\r]", stdout.decode("utf-8"))
            assert any(line.startswith("Testing DataLoader 0: 100%") for line in output_lines)
