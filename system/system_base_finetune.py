import torch
from pytorch_lightning.metrics.functional import accuracy

from system import system_abstract


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

        if isinstance(self.hparams.dataset, list) and len(
                self.hparams.dataset) == 1:
            self.hparams.dataset = self.hparams.dataset[0]
        self._create_model(self.num_classes)

    def forward(self, x):
        out = self.feature.forward(x)
        if self.hparams.use_norm:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        scores = self.classifier.forward(out)
        return scores, out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="base_ft")
        parser.add_argument("--lr", type=float, default=0.02)
        parser.add_argument("--batch_size", type=int, default=64)

        parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser.add_argument("--max_epochs", type=int, default=200)
        parser.add_argument("--fine_tune_type", type=str, default="LR")
        parser.add_argument("--warm_epochs", type=int, default=5)
        parser.add_argument("--warm_up",
                            type=str,
                            default='true',
                            choices=['true', 'false'],
                            help="whether to lr warm-up")
        parser.add_argument("--sync_batchnorm", action="store_true")
        # optimizer
        parser.add_argument("--optimizer", type=str, default="SGD")
        parser.add_argument("--scheduler", type=str, default='cosine')
        return parser

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, tuple) or isinstance(x, list) and len(x) == 2:
            x = x[0]

        scores, _ = self(x)
        _, predicted = torch.max(scores.data, 1)
        loss = torch.nn.functional.cross_entropy(scores, y)

        with torch.no_grad():
            accur = accuracy(predicted.detach(), y)

        tqdm_dict = {"loss_train": loss, "top1.val": accur}
        self.log_dict(tqdm_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # -------------------------------- validation -------------------------------- #
    def validation_step(self, *args, **kwargs):
        if self.hparams.eval_mode == 'few_shot':
            return super().validation_step(*args, **kwargs)
        elif self.hparams.eval_mode == 'linear':
            return self._linear_validation_step(*args, **kwargs)

    def validation_epoch_end(self, outputs):
        if self.hparams.eval_mode == 'few_shot':
            return super().validation_epoch_end(outputs)
        elif self.hparams.eval_mode == 'linear':
            return self._linear_validation_epoch_end(outputs)

    def get_feature_extractor(self):
        """ return feature extractor """
        return self.feature