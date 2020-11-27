from typing import Dict, List, Optional

from pytorch_lightning.callbacks.base import Callback


class LearningRateLogger(Callback):
    def __init__(self, logging_interval: Optional[str] = None):
        self.logging_interval = logging_interval
        self.names = ['learning_rate']
        self.lrs = {name: [] for name in self.names}

    def on_train_batch_start(self, trainer, *args, **kwargs):
        latest_stat = self._extract_lr(trainer, 'step')

        if trainer.logger is not None and latest_stat:
            trainer.logger.experiment.log(latest_stat)

    def _extract_lr(self, trainer, interval: str) -> Dict[str, float]:
        latest_stat = {}

        for name, scheduler in zip(self.names, trainer.lr_schedulers):
            if scheduler['interval'] == interval or interval == 'any':
                param_groups = scheduler['scheduler'].optimizer.param_groups
                if len(param_groups) != 1:
                    for i, pg in enumerate(param_groups):
                        lr, key = pg['lr'], f'{name}/pg{i + 1}'
                        self.lrs[key].append(lr)
                        latest_stat[key] = lr
                else:
                    self.lrs[name].append(param_groups[0]['lr'])
                    latest_stat[name] = param_groups[0]['lr']

        return latest_stat
