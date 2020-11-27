from typing import Dict, List, Optional

from pytorch_lightning.callbacks.base import Callback


class LearningRateLogger(Callback):
    def __init__(self, logging_interval: Optional[str] = None):
        self.logging_interval = logging_interval
        self.name = 'learning_rate'

    def on_train_batch_start(self, trainer, *args, **kwargs):
        latest_lr = self._extract_lr(trainer, 'step')

        if trainer.logger is not None and latest_lr:
            trainer.logger.experiment.log(latest_lr)

    def _extract_lr(self, trainer, interval: str) -> Dict[str, float]:
        latest_lr = {}

        for scheduler in trainer.lr_schedulers:
            if scheduler['interval'] == interval or interval == 'any':
                param_groups = scheduler['scheduler'].optimizer.param_groups
                if len(param_groups) != 1:
                    for i, pg in enumerate(param_groups):
                        lr, key = pg['lr'], f'{self.name}/pg{i + 1}'
                        latest_lr[key] = lr
                else:
                    latest_lr[self.name] = param_groups[0]['lr']

        return latest_lr
