import numpy as np
from pytorch_lightning.loggers.base import rank_zero_experiment
import torch.distributed as dist
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import sys
import os
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from colorama import init, Fore
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.step_result import recursive_gather, recursive_stack

init(autoreset=True)


class LatestCheckpoint(Callback):
    """ save latest checkpoint """
    def __init__(self, ckpt_path, period=1, verbose=False, show_epoch=False):
        super().__init__()
        self.period = period
        self.ckpt_path = ckpt_path
        self.verbose = verbose
        self.show_epoch = show_epoch

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.period == 0:
            os.makedirs(self.ckpt_path, exist_ok=True)
            path_to_save = os.path.join(self.ckpt_path, "last.ckpt")
            trainer.save_checkpoint(path_to_save)

            if (self.show_epoch is not None
                    and trainer.current_epoch % self.show_epoch == 0
                    and trainer.current_epoch > 0):
                path_to_save = os.path.join(
                    self.ckpt_path, f"last_epoch_{trainer.current_epoch}.ckpt")
                trainer.save_checkpoint(path_to_save)

            if self.verbose:
                logger.debug(f"SAVE CHECKPOINT : {path_to_save}")


class CustomLogger(LightningLoggerBase):
    def __init__(self, config):
        super().__init__()
        self._logger = logger
        self.config = config
        self._name = self.config.model_name
        self._save_dir = config.log_path
        self._version = None
        if config.test:
            self._save_dir = "test_" + config.log_path
        self._experiment = None

    # @rank_zero_only
    def _create_logger(self):
        # CLI logger
        self._logger.remove()
        self._logger.configure(handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=''),
                level='DEBUG',
                colorize=True,
                format=
                "<green>{time: MM-DD at HH:mm}</green>  <level>{message}</level>",
                enqueue=True),
        ])

        # add file handler for training mode
        if self.config.disable_logfile == 'false':
            os.makedirs(self.log_dir, exist_ok=True)
            logfile = os.path.join(self.log_dir, "log.txt")
            self._logger.info(f"Log to file {logfile}")
            self._logger.add(sink=logfile,
                             mode='w',
                             format="{time: MM-DD at HH:mm} | {message}",
                             level="DEBUG",
                             enqueue=True)

        # enable tensorboard logger
        self.tb_log = self.config.tb_log
        if self.config.tb_log:
            self.tb_logger = TensorBoardLogger(os.path.join(
                self.log_dir, "tb_logs"),
                                               name=self.name)

    @property
    def log_dir(self):
        version = self.version if isinstance(
            self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment:
            return self._experiment
        self._create_logger()
        self._experiment = self._logger
        return self._experiment

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            try:
                return value.item()
            except ValueError:
                return value.mean().item()
        return value

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if len(metrics) == 0:
            return
        if self.tb_log:
            self.tb_logger.log_metrics(metrics, step)

        metrics_str = "  ".join([
            f"{k}: {self._handle_value(v):<4.4f}" for k, v in metrics.items()
            if k != 'epoch'
        ])

        if metrics_str.strip() == '':
            return

        if step is not None:
            metrics_str = f"step: {step:<6d} :: " + metrics_str
        if 'epoch' in metrics:
            metrics_str = f"epoch: {int(metrics['epoch']):<4d}  " + metrics_str
        self.experiment.info(metrics_str)

    @rank_zero_only
    def info_metrics(self, metrics, epoch=None, step=None, level='INFO'):
        if isinstance(metrics, str):
            self.experiment.info(metrics)
            return

        _str = ""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            _str += f"{k}= {v:<4.4f}  "
        self.experiment.log(level,
                            f"epoch {epoch: <4d}: step {step:<6d}:: {_str}")

    @rank_zero_only
    def log(self, msg, level='DEBUG'):
        self.experiment.log(level, msg)

    @property
    def name(self):
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            logger.warning('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir,
                                          d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_hyperparams(self, params):
        _str = ""
        for k in sorted(params):
            v = params[k]
            _str += Fore.LIGHTCYAN_EX + str(k) + "="
            _str += Fore.WHITE + str(v) + ", "
        self.experiment.info("\nhyper-parameters:\n" + _str)
        return

    @property
    def root_dir(self) -> str:
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def save_dir(self):
        return self._save_dir


class AverageMeter():
    def __init__(self, use_ddp=False):
        super().__init__()
        self.reset()
        self.use_ddp = use_ddp
        self.best_val = -1

    def add(
            self,
            value,
            n=1,
    ):
        if self.use_ddp is False:
            self.val = value
            self.sum += value
            self.var += value * value
            self.n += n
        else:
            var = value**2
            dist.all_reduce(value)
            dist.all_reduce(var)
            self.sum += value
            self.var += var
            self.n += n * dist.get_world_size()
            self.val = value / dist.get_world_size()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0

    @property
    def value(self):
        return self.val

    @property
    def mean(self):
        return (self.sum / self.n)

    @property
    def avg(self):
        return self.mean

    @property
    def max(self):
        mean_val = self.mean
        if self.use_ddp:
            dist.all_reduce(mean_val)
            mean_val /= dist.get_world_size()

        if mean_val > self.best_val:
            self.best_val = mean_val

        return self.best_val

    @property
    def std(self):
        acc_std = torch.sqrt(self.var / self.n - self.mean**2)
        return 1.96 * acc_std / np.sqrt(self.n)


def set_bn_to_eval(mod):
    for m in mod.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def aggregate_dict(inputs):
    results = recursive_gather(inputs, {})
    recursive_stack(results)
    return results


@torch.no_grad()
def concat_all_ddp(tensor):
    try:
        world_size = torch.distributed.get_world_size()
    except AssertionError:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# moco utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def moco_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Classifier_d(nn.Module):
    def __init__(self, dim, dim_hidden, n_way, drop_p=0, apply_bn='false'):
        super().__init__()
        self.n_way = n_way
        if dim_hidden is None:
            dim_hidden = dim

        def _get_layer():
            if apply_bn == 'false':
                return nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU(),
                                     nn.Linear(dim_hidden, n_way))
            else:
                return nn.Sequential(nn.Linear(dim, dim_hidden, bias=False),
                                     nn.BatchNorm1d(dim_hidden), nn.ReLU(),
                                     nn.Linear(dim_hidden, n_way))

        self.dropout = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
        self.lin1 = _get_layer()
        self.lin2 = _get_layer()

    def forward(self, x):
        x = self.dropout(x)
        x1 = F.normalize(self.lin1(x), dim=-1)
        x2 = F.normalize(self.lin2(x), dim=-1)
        return torch.cat((x1, x2), dim=-1)
