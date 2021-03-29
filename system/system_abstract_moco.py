import abc
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

from system import system_abstract


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)
        self.feature = self.base_encoder()
        self.feature_k = self.base_encoder()
        self.use_ddp = False
        self._create_moco(self.feature, self.feature_k)

        if (not self.hparams.disable_validation
                and self.hparams.eval_mode == 'linear'):
            if self.hparams.build_linear_evaluator == "true":
                self._build_linear_evaluator()
            self.feature_extractor = self.get_feature_extractor()

    # def setup(self, stage):
    #     super().setup(stage)
    #     if stage ==

    def forward(self, x):
        out = self.feature.forward(x)
        if self.hparams.use_norm:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def on_train_start(self):
        # self.use_ddp = self.trainer.use_ddp
        if not hasattr(self, "use_ddp"): self.use_ddp = self.trainer.use_ddp
        self.moco.use_ddp = self.use_ddp

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="moco_ss")
        parser.add_argument("--lr", type=float, default=0.02)
        parser.add_argument("--batch_size", type=int, default=64)

        parser.add_argument("--check_val_every_n_epoch", type=int, default=20)
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--moco_dim_mlp", type=int, default=128)

        parser.add_argument("--max_epochs", type=int, default=400)
        parser.add_argument('--dataset',
                            default='miniImageNet_train',
                            type=str,
                            help='training base model')

        parser.add_argument('--train_aug', type=str, default='MoCo')

        parser.add_argument("--eval_mode", type=str, default="linear")

        # moco
        parser.add_argument("--optimizer", type=str, default="SGD")
        parser.add_argument("--load_base",
                            action="store_true",
                            help="load only base")
        parser.add_argument("--gpus", type=str, default="-1")
        parser.add_argument("--moco_mlp",
                            default='true',
                            choices=['true', 'false'])
        parser.add_argument("--moco_K", type=int, default=64000)
        parser.add_argument("--moco_T", type=float, default=0.07)

        parser.add_argument("--moco_dim_hidden", type=int, default=None)
        parser.add_argument("--moco_dim_final", type=int, default=128)
        parser.add_argument("--moco_mlp_bn",
                            type=str,
                            default='false',
                            choices=['true', 'false'])
        parser.add_argument("--moco_mlp_to_feature", type=int, default=0)
        parser.add_argument("--lr_linear_multiplier", type=float, default=10)
        parser.add_argument("--lr_linear_momentum", type=float, default=0)

        return parser

    # --------------------------------- training --------------------------------- #
    @abc.abstractmethod
    def _training_step(self, *args, **kwargs):
        """ Implement your own step """

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            # self.linear_evaluator.requires_grad_(False)
            # self.moco.requires_grad_(True)
            return self._training_step(batch, batch_idx)
        else:
            # self.linear_evaluator.requires_grad_(True)
            # self.moco.requires_grad_(False)
            return self._linear_training_step(batch, batch_idx)

    # ----------------------------------- model ---------------------------------- #
    @abc.abstractmethod
    def _create_moco(self, enc_q, enc_k):
        """ create moco model """

    # --------------------------------- optimizer -------------------------------- #
    def configure_optimizers(self):
        if self.hparams.eval_mode != 'linear' or self.hparams.disable_validation:
            return super().configure_optimizers()
        else:
            # add a linear evaluation head
            self.linear_evaluator.requires_grad_(False)
            base_optimizer = super().configure_optimizers()
            # self.linear_evaluator.requires_grad_(True)
            linear_optimizer = torch.optim.SGD(
                self.linear_evaluator.parameters(),
                lr=self.hparams.lr * self.hparams.lr_linear_multiplier,
                momentum=self.hparams.lr_linear_momentum,
                weight_decay=0)
            return base_optimizer, linear_optimizer

    # -------------------------- linear evaluation head -------------------------- #
    def _get_representations(self, x):
        if len(x) == 2 and (isinstance(x, list) or isinstance(x, tuple)):
            x = x[0]
        representations = self.feature_extractor.forward(x)
        return representations

    def _build_linear_evaluator(self, drop_p=0.2):
        drop = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
        fc = nn.Linear(self.feature.final_feat_dim, self.num_classes)
        if self.hparams.linear_bn:
            self.linear_evaluator = nn.Sequential(
                nn.BatchNorm1d(self.feature.final_feat_dim, affine=False),
                drop, fc)
        else:
            self.linear_evaluator = nn.Sequential(drop, fc)

    def _forward_batch(self, x):
        with torch.no_grad():
            representations = self._get_representations(x)
        mlp_preds = self.linear_evaluator(representations)
        return mlp_preds, representations

    def _linear_training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        mlp_preds, _ = self._forward_batch(x)
        mlp_loss = F.cross_entropy(mlp_preds, y)
        preds = torch.max(mlp_preds, dim=-1)[1]
        try:
            acc = accuracy(preds, y)
        except RuntimeError:
            acc = torch.tensor(0., device=self.device, type=x.dtype)
        self.log_dict({
            'ft_loss': mlp_loss,
            'ft_acc': acc
        },
                      logger=True,
                      on_step=True,
                      on_epoch=True)
        return mlp_loss