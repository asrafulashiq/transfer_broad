import os
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
import itertools
from scipy.stats import rankdata


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

        self.perturbation = [
            'gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
            'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale',
            'speckle_noise', 'gaussian_blur', 'snow', 'shear'
        ]
        self.perturbation = ['gaussian_noise', 'shot_noise']
        self.difficulty = [1]  #[1, 2, 3]

        self.list_errors = {k: torch.zeros(5) for k in self.perturbation}

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

        parser.add_argument("--dataset", default=None)
        parser.add_argument("--data_root",
                            type=str,
                            default="~/datasets/imagenet-p")
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

        result = {""}
        result["gt"] = y
        result["prob"] = scores

    def validation_epoch_end(self, outputs_all):
        for i, outputs in enumerate(outputs_all):
            outputs = utils_system.aggregate_dict(outputs)
            epoch_probs = concat_all_ddp(outputs.prob)
            epoch_gt = concat_all_ddp(outputs.gt)

            epoch_preds = torch.argmax(epoch_probs, dim=-1)

            predictions = epoch_preds.data.cpu().numpy()
            ranks = np.asarray([
                np.uint16(rankdata(-frame, method='ordinal'))
                for frame in epoch_probs.data.cpu().numpy()
            ])

            _accuracy = accuracy(epoch_preds, epoch_gt)
            error = 1 - _accuracy

            self.list_errors[self.list_type[i][0]][self.list_type[i][1] -
                                                   1] = error

        error_rates = []
        for i, (k, v) in enumerate(self.list_errors.items()):
            _err = v.mean() * 100
            self.logger.experiment.info(
                f"Distortion {k:15s} | CE : {_err:.2f}% ")
            error_rates.append(_err)
        mCE = torch.stack(error_rates, dim=-1).mean()
        self.logger.experiment.info(f"mCE :: {mCE}%")

        self.log("mCE", mCE)

    # --------------------------------- val data --------------------------------- #
    def prepare_data(self) -> None:
        pass

    def val_dataloader(self):
        tsfm = transforms.Compose([
            transforms.CenterCrop(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        list_loaders = []
        self.list_type = list(
            itertools.product(self.perturbation, self.difficulty))

        for perturbation, difficulty in tqdm(
                self.list_type, desc="loading distorted dataset ..",
                leave=False):
            if difficulty <= 1:
                root = os.path.join(self.hparams.data_root, perturbation,
                                    str(difficulty))
            else:
                raise NotImplementedError
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

            list_loaders.append(distorted_loader)
        return list_loaders

    def test_dataloader(self):
        return self.val_dataloader()

    def train_dataloader(self):
        raise NotImplementedError("Should run in `test` mode")


def flip_prob(predictions, noise_perturbation=False, difficulty=1):
    result = 0
    step_size = 1 if noise_perturbation else difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# def ranking_dist(
#         ranks,
#         noise_perturbation=True if 'noise' in args.perturbation else False,
#         mode='top5'):
#     result = 0
#     step_size = 1 if noise_perturbation else args.difficulty

#     for vid_ranks in ranks:
#         result_for_vid = []

#         for i in range(step_size):
#             perm1 = vid_ranks[i]
#             perm1_inv = np.argsort(perm1)

#             for rank in vid_ranks[i::step_size][1:]:
#                 perm2 = rank
#                 result_for_vid.append(dist(perm2[perm1_inv], mode))
#                 if not noise_perturbation:
#                     perm1 = perm2
#                     perm1_inv = np.argsort(perm1)

#         result += np.mean(result_for_vid) / len(ranks)

#     return result