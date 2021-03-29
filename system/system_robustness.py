import os
from torch import nn
import torchvision
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from system import system_abstract
from .utils_system import concat_all_ddp
from system import utils_system
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.calibration_tools import show_calibration_results


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)
        self.dm = datamodule
        self.conf_data = self.dm.conf_data

        self._create_model(self.num_classes)
        self.drop_fn = torch.nn.Dropout(
            self.hparams.linear_drop
        ) if self.hparams.linear_drop > 0 else torch.nn.Identity()

        self.indices_in_1k = np.load(
            f"data_loader/ind_rob_imagenet{self.hparams.rob_type}.npy",
            allow_pickle=True)

        # FIXME had to add this for some version issue
        if self.hparams.ckpt is not None and "base_plus_moco" in self.hparams.ckpt:
            self.base_head = self.classifier

    def load_base(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = torch.load(
                ckpt_path,
                map_location=lambda storage, loc: storage)['state_dict']
            new_state = {}
            for k, v in ckpt.items():
                if 'feature.' in k:
                    new_state[k.replace('feature.', '')] = v
            self.feature.load_state_dict(new_state,
                                         strict=not self.hparams.load_flexible)

    def forward(self, x):
        out = self.feature.forward(x)
        out = self.drop_fn(out)
        scores = self.classifier.forward(out)
        return scores, out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         conflict_handler="resolve")
        parser.add_argument("--model_name", type=str, default="linear_eval")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--eval_mode", type=str, default="linear")

        parser.add_argument("--pretrained", action="store_true")

        parser.add_argument("--num_classes", type=int, default=1000)
        parser.add_argument("--rob_type",
                            type=str,
                            default="a",
                            choices=['a', 'o', 'r'])

        parser.add_argument("--dataset", default=None)
        parser.add_argument("--data_root", type=str, default="~/datasets")
        parser.add_argument('--val_dataset',
                            default=None,
                            type=str,
                            help='val dataset to use')
        parser.add_argument('--val_aug', type=str, default='false')
        parser.add_argument("--linear_drop", type=float, default=0)
        parser.add_argument("--num_workers", type=int, default=4)
        return parser

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    # -------------------------------- validation -------------------------------- #
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        scores, _ = self(x)
        scores = scores[..., self.indices_in_1k]

        result = {}
        result["gt"] = y
        result["prob"] = scores.softmax(dim=-1)
        self.log("batch", batch_idx, on_step=True, on_epoch=False)
        return result

    def validation_epoch_end(self, outputs):
        to_np = lambda x: x.data.to('cpu').numpy()

        outputs = utils_system.aggregate_dict(outputs)
        epoch_probs = concat_all_ddp(outputs["prob"])
        epoch_gt = concat_all_ddp(outputs["gt"])
        confidence, epoch_preds = torch.max(epoch_probs, dim=-1)

        _accuracy = accuracy(epoch_preds, epoch_gt)
        correct = epoch_preds.eq(epoch_gt)
        calib, aurra, _ = show_calibration_results(to_np(confidence),
                                                   to_np(correct))

        self.log("calib", calib)
        self.log("aurra", aurra)
        self.log("accuracy", _accuracy)
        self.log("error", 1 - _accuracy)

    # --------------------------------- val data --------------------------------- #
    def prepare_data(self) -> None:
        pass

    def val_dataloader(self):
        tsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        root = os.path.join(self.hparams.data_root,
                            f"imagenet-{self.hparams.rob_type}")
        distorted_dset = torchvision.datasets.ImageFolder(root=root,
                                                          transform=tsfm)
        distorted_loader = torch.utils.data.DataLoader(
            distorted_dset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        if self.use_ddp:
            self.trainer.replace_sampler_ddp = True
            distorted_loader = self.trainer.auto_add_sampler(
                distorted_loader, True)
            self.trainer.replace_sampler_ddp = False
        self.logger.experiment.info(
            f"# imagenet-{self.hparams.rob_type}: {len(distorted_dset)}")
        return distorted_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def train_dataloader(self):
        raise NotImplementedError("Should run in `test` mode")
