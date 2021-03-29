import math
from typing import Optional
from system import utils_system
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from system import system_abstract
from .utils_system import concat_all_ddp
import uncertainty_metrics.numpy as um
import torchvision
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)
        self.dm = datamodule
        self.conf_data = self.dm.conf_data
        self._create_model(None)

        self.logreg = LogisticRegression(C=self.hparams.l2_C,
                                         verbose=1,
                                         random_state=self.hparams.seed,
                                         max_iter=self.hparams.max_iter)

        self.feature.requires_grad_(False)
        self.feature.eval()

    def forward(self, x):
        if len(x) == 2 and (isinstance(x, list) or isinstance(x, tuple)):
            x = x[0]
        out = self.feature.forward(x)
        if self.hparams.use_norm:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="linear_eval")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--eval_mode", type=str, default="linear")

        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--max_epochs", type=int, default=1)

        parser.add_argument("--num_classes", type=int, default=None)

        parser.add_argument("--dataset", default='CIFAR100_train', type=str)
        parser.add_argument('--val_dataset',
                            default='CIFAR100_test',
                            type=str,
                            help='val dataset to use')

        parser.add_argument('--train_aug', type=str, default='false')
        parser.add_argument('--val_aug', type=str, default='false')
        parser.add_argument(
            "--unfreeze_lastk",
            type=int,
            default=0,
            help="unfreeze k layers (default:0: all unfreeze, 0:no_unfreeze)")
        parser.add_argument("--gpus", type=str, default="0")

        # NOTE disable automatic optimization
        parser.add_argument("--automatic_optimization", action="store_true")
        parser.add_argument("--l2_C", type=float, default=1.0)
        parser.add_argument("--max_iter", type=int, default=200)
        return parser

    # --------------------------------- training --------------------------------- #
    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

    def training_step(self, batch, batch_idx):
        self.feature.requires_grad_(False)
        self.feature.eval()
        with torch.no_grad():
            assert self.feature.training == False
            x, y = batch
            features = self(x).flatten(1)
            features_np = to_numpy(features)

            self.x_train.extend([ft for ft in features_np])
            self.y_train.extend([int(yt) for yt in y])

            tqdm_dict = {"itr": self.global_step}
            self.log_dict(tqdm_dict, on_step=True)

    def on_validation_epoch_start(self) -> None:
        self.logreg.fit(self.x_train, self.y_train)
        self.logger.experiment.info("Fitted LOGISTIC_REGRESSION")

    def validation_step(self, batch, batch_idx):
        self.feature.eval()
        with torch.no_grad():
            assert self.feature.training == False
            x, y = batch
            features = self(x).flatten(1)
            features_np = to_numpy(features)

            self.x_test.extend([ft for ft in features_np])
            self.y_test.extend([int(yt) for yt in y])

            tqdm_dict = {"val_itr": batch_idx}
            self.log_dict(tqdm_dict)

    def validation_epoch_end(self, outputs):

        mean_accuracy = self.logreg.score(self.x_test, self.y_test)
        mean_accuracy = torch.tensor(mean_accuracy).to(self.device)

        self.logger.log(f"mean_accuracy: {mean_accuracy: .4f}", "CRITICAL")

        info = {
            "acc_mean": mean_accuracy,
        }
        self.log_dict(info, logger=True, prog_bar=True)

        self.logger.log_csv(info, step=self.global_step)

    # --------------------------------- optimizer -------------------------------- #
    def configure_optimizers(self):
        xx = torch.nn.Linear(3, 3, bias=True)
        optimizer = torch.optim.SGD(xx.parameters(), lr=1e-2)
        return optimizer

    # --------------------------------- val data --------------------------------- #
    def prepare_data(self):
        self.dm.prepare_data()

    def val_dataloader(self):
        if self.hparams.val_as_test == 'true':
            return self.dm.val_dataloader_val_split(pl_trainer=self.trainer,
                                                    use_ddp=self.use_ddp)
        base_loader = self.dm.get_simple_dataloader(self.hparams.val_dataset,
                                                    aug=self.hparams.val_aug,
                                                    pl_trainer=self.trainer,
                                                    use_ddp=self.use_ddp,
                                                    opt=self.hparams,
                                                    drop_last=False,
                                                    shuffle=False)
        return base_loader

    def test_dataloader(self):
        return self.val_dataloader()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        return x.data.numpy()
    else:
        return x
