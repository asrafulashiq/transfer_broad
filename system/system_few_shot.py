from system import system_abstract
import torch


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
        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="fewshot")
        parser.add_argument("--fine_tune_type", type=str, default="LR")
        parser.add_argument("--eval_mode", default="few_shot")
        parser.add_argument("--val_aug", default="few_shot_test")
        parser.add_argument("--num_episodes", type=int, default=600)
        parser.add_argument("--gpus", default="1")
        parser.add_argument("--num_classes", type=int, default=1000)
        # parser.add_argument("--test", action="store_false")
        parser.add_argument("--dataset", default=None)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--split_fraction", type=float, default=0.7)
        parser.add_argument("--gpus", type=str, default="1")
        # parser.add_argument("--load_base", action="store_false")
        return parser

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_feature_extractor(self):
        """ return feature extractor """
        return self.feature

    def train_dataloader(self):
        return None
