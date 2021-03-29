import numpy as np
import pytorch_lightning as pl
import torch
from data_loader import utils_data
import os
from pytorch_lightning.utilities.distributed import rank_zero_only

from omegaconf import OmegaConf
from loguru import logger
from .utils_data import get_split


class DataModule(object):
    def __init__(self, hparams, conf_path=None) -> None:
        self.hparams = hparams
        if conf_path is None:
            conf_path = "data_loader/config_path.yaml"
        self.conf_data = OmegaConf.load(conf_path)
        for key in self.conf_data:
            self.conf_data[key]['data_path'] = os.path.expanduser(
                self.conf_data[key]['data_path'])

    def get_path(self, dname):
        basename = dname.split("_")[0]  # check if there is any split
        return self.conf_data[basename]['data_path']

    def get_num_class(self, dname):
        basename = dname.split("_")[0]  # check if there is any split
        return self.conf_data[basename]['num_class']

    def prepare_data(self, *args, **kwargs):
        """
        If necessary 
            - download datasets
            - prepare indices and store into cache
        """
        def fn_prepare(_dataset):
            if isinstance(_dataset, list) or isinstance(_dataset, tuple):
                for dname in _dataset:
                    suff = get_split(dname)[1]
                    if suff:
                        utils_data.prepare_data_indices(dname,
                                                        self.get_path(dname),
                                                        opt=self.hparams)
            elif isinstance(_dataset, str):
                suff = get_split(_dataset)[1]
                if suff:
                    utils_data.prepare_data_indices(_dataset,
                                                    self.get_path(_dataset),
                                                    opt=self.hparams)
            else:
                pass

        if self.hparams.dataset:
            fn_prepare(self.hparams.dataset)

        if not self.hparams.disable_validation and self.hparams.val_dataset:
            fn_prepare(self.hparams.val_dataset)

    def train_dataloader(self,
                         pl_trainer=None,
                         use_ddp=False,
                         *args,
                         **kwargs):
        """ get train dataloader """
        if self.hparams.dataset is None:
            self.hparams.dataset = self.hparams.val_dataset
        datamgr = utils_data.SimpleDataManager(
            self.hparams.image_size,
            batch_size=self.hparams.batch_size,
            dataset_name=self.hparams.dataset,
            unlabel=self.hparams.benchmark,
            opt=self.hparams)

        if isinstance(self.hparams.dataset, list) or isinstance(
                self.hparams.dataset, tuple):
            data_path = [self.get_path(_dat) for _dat in self.hparams.dataset]
        else:
            data_path = self.get_path(self.hparams.dataset)
        base_loader = datamgr.get_data_loader(
            data_path,
            aug=self.hparams.train_aug,
            consecutive_label=True,
            limit_data=self.hparams.limit_train_data)

        if use_ddp:
            pl_trainer.replace_sampler_ddp = True
            base_loader = pl_trainer.auto_add_sampler(base_loader, True)
            pl_trainer.replace_sampler_ddp = False
            self._log(f"use_ddp, len: {len(base_loader)}")
        self._log(f"#(train_dataloader batches):  {len(base_loader)}")

        return base_loader

    @rank_zero_only
    def _log(self, msg):
        logger.info(msg)

    def test_dataloader(self, *args, **kwargs):
        return self.val_dataloader(*args, **kwargs)

    def get_fewshot_dataloader(self,
                               pl_trainer=None,
                               use_ddp=False,
                               *args,
                               **kwargs):
        """ return few-shot data-loader """
        datasets = self.hparams.val_dataset
        if datasets is None or datasets == 'none':
            return None

        if not isinstance(self.hparams.val_dataset, list):
            datasets = [self.hparams.val_dataset]

        list_loader = []
        for dset in datasets:
            few_shot_params = dict(n_way=self.hparams.test_n_way,
                                   n_support=self.hparams.n_shot)
            if use_ddp:
                ddp_args = dict(num_replicas=pl_trainer.num_nodes *
                                pl_trainer.num_processes,
                                rank=pl_trainer.global_rank)
            else:
                ddp_args = {}
            datamgr = utils_data.SetDataManager(
                self.get_path(dset),
                self.get_num_class(dset),
                self.hparams.image_size,
                n_episode=self.hparams.num_episodes,
                n_query=self.hparams.n_query,
                dataset_name=dset,
                opt=self.hparams,
                **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug=self.hparams.val_aug,
                                                   use_ddp=use_ddp,
                                                   dist_args=ddp_args)
            list_loader.append(novel_loader)

        return list_loader

    def get_simple_dataloader(self,
                              dataset_name,
                              aug='true',
                              pl_trainer=None,
                              use_ddp=False,
                              opt=None,
                              drop_last=True,
                              shuffle=True):
        """ get dataloader for simple evaluation (linear/finetune) """
        datamgr = utils_data.SimpleDataManager(
            self.hparams.image_size,
            batch_size=self.hparams.batch_size,
            dataset_name=dataset_name,
            opt=opt)

        if isinstance(dataset_name, list):
            data_path = [self.get_path(_dat) for _dat in dataset_name]
        else:
            data_path = self.get_path(dataset_name)

        base_loader = datamgr.get_data_loader(data_path,
                                              aug=aug,
                                              consecutive_label=True,
                                              drop_last=drop_last,
                                              shuffle=shuffle)
        if use_ddp:
            pl_trainer.replace_sampler_ddp = True
            base_loader = pl_trainer.auto_add_sampler(base_loader, True)
            pl_trainer.replace_sampler_ddp = False

        return base_loader

    def train_dataloader_val_split(self,
                                   pl_trainer=None,
                                   use_ddp=False,
                                   *args,
                                   **kwargs):
        """ split train data-loader into train/val split, return train """
        _loader = self.train_dataloader(pl_trainer, False, *args, **kwargs)
        dataset = _loader.dataset

        sub_dataset = get_random_split(
            dataset,
            trn_percent=self.hparams.train_val_split,
            training=True,
            seed=self.hparams.seed)

        base_loader = torch.utils.data.DataLoader(
            sub_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=True)
        if use_ddp:
            pl_trainer.replace_sampler_ddp = True
            base_loader = pl_trainer.auto_add_sampler(base_loader, True)
            pl_trainer.replace_sampler_ddp = False
        self._log(
            f"#(train_dataloader_val_split batches):  {len(base_loader)}")

        return base_loader

    def val_dataloader_val_split(self,
                                 pl_trainer=None,
                                 use_ddp=False,
                                 *args,
                                 **kwargs):
        """ split train data-loader into train/val split, return val """
        _loader = self.train_dataloader(pl_trainer, False, *args, **kwargs)
        dataset = _loader.dataset

        sub_dataset = get_random_split(
            dataset,
            trn_percent=self.hparams.train_val_split,
            training=False,
            seed=self.hparams.seed)

        base_loader = torch.utils.data.DataLoader(
            sub_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=True)
        if use_ddp:
            pl_trainer.replace_sampler_ddp = True
            base_loader = pl_trainer.auto_add_sampler(base_loader, True)
            pl_trainer.replace_sampler_ddp = False

        self._log(f"#(val_dataloader_val_split batches): {len(base_loader)}")
        return base_loader


def get_random_split(dataset, trn_percent: float, training=True, seed=0):
    """ split dataset into train/val index
        Args:
            trn_percent: percentage for training 
    """
    rng = np.random.RandomState(seed)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)

    trn_len = int(len(dataset) * trn_percent)
    val_len = len(dataset) - trn_len

    trn_indices = all_idx[:trn_len]
    val_indices = all_idx[trn_len:]

    if training:
        return torch.utils.data.Subset(dataset, trn_indices)
    else:
        return torch.utils.data.Subset(dataset, val_indices)
