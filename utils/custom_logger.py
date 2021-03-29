from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.loggers import LightningLoggerBase
from loguru import logger
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import os
from tqdm import tqdm
from colorama import init, Fore
from pytorch_lightning.loggers.csv_logs import ExperimentWriter

init(autoreset=True)


class CustomLogger(LightningLoggerBase):
    def __init__(self, config):
        super().__init__()
        self._logger = logger
        self.config = config
        self._name = self.config.model_name
        self._save_dir = config.log_path
        self._version = config.version
        if config.test:
            self._save_dir = "test_" + config.log_path
        self._experiment = None

        self.csv_logger = None

    def _create_logger(self):
        # CLI logger
        self._logger.remove()
        self._logger.configure(handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=''),
                level='DEBUG',
                colorize=True,
                format=
                "<green>{time: MM-DD at HH:mm}</green> <level>{message}</level>",
                enqueue=True),
        ])

        # add file handler for training mode
        if self.config.disable_logfile == 'false':
            os.makedirs(self.log_dir, exist_ok=True)
            logfile = os.path.join(self.log_dir, "log.txt")
            self._logger.warning(f"Log to file {logfile}")
            self._logger.add(sink=logfile,
                             mode='w',
                             format="{time: MM-DD at HH:mm} | {message}",
                             level="DEBUG",
                             enqueue=True)
            self._logger.info(f"log to {logfile}")

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
        # if self.tb_log:
        #     self.tb_logger.log_metrics(metrics, step)

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
    def log_csv(self, metrics, step=None):
        if self.csv_logger is None:
            self.csv_logger = CSVWriter(self.log_dir)
        self.csv_logger.log_metrics(metrics, step)

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

    def dict_to_str(self, d, parent=""):
        import collections
        if not isinstance(d, collections.abc.MutableMapping):
            return str(d)
        _str = ""
        for k in sorted(d):
            v = d[k]
            if parent:
                k = parent + "." + k
            v = self.dict_to_str(v, parent=k)

            _str += Fore.LIGHTCYAN_EX + str(k) + "="
            _str += Fore.WHITE + str(v) + ", "
        return _str

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


import csv
import io
from typing import Dict, Optional
import logging


class CSVWriter():
    NAME_METRICS_FILE = 'metrics.csv'

    def __init__(self, log_dir: str) -> None:
        self.metrics = []

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir,
                                              self.NAME_METRICS_FILE)

        # create logger on the current module and set its level
        self.logger = logging.getLogger("metrics_csv")
        self.logger.setLevel(logging.INFO)
        self.needs_header = True

        # create a channel for handling the logger (stderr) and set its format
        ch = logging.FileHandler(self.metrics_file_path, mode='w')
        # connect the logger to the channel
        self.logger.addHandler(ch)

    def log_metrics(self,
                    metrics_dict: Dict[str, float],
                    step: Optional[int] = None) -> None:
        """Record metrics"""
        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics['step'] = step
        self.metrics.append(metrics)

        if self.needs_header:
            self.logger.info(','.join(sorted(metrics)))
            self.needs_header = False

        self.logger.info(','.join([str(metrics[k]) for k in sorted(metrics)]))

    @property
    def file_path(self):
        return self.metrics_file_path
