import argparse
import torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import abc
import math
import torchvision
import torchvision.models.resnet as resnet_models
from sklearn.metrics import classification_report, accuracy_score

import system.utils_system as utils_system
# import backbone
from .few_shot_mixin import FewShotMixin
from .linear_mixin import LinearMixin, calc_ece


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(pl.LightningModule, FewShotMixin, LinearMixin):
    """ Abstract class """
    def __init__(self, hparams, datamodule=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dm = datamodule

        if self.hparams.num_classes is None and self.dm is not None:
            self.num_classes = self._cal_num_classes(self.hparams.dataset)
        else:
            self.num_classes = self.hparams.num_classes

        self.val_names = self.hparams.val_dataset if isinstance(
            self.hparams.val_dataset, list) else [self.hparams.val_dataset]

    def _cal_num_classes(self, dataset_name):
        def fn_num(dset):
            return self.dm.get_num_class(dset)

        if isinstance(dataset_name, str):
            return fn_num(dataset_name)
        elif isinstance(dataset_name, list):
            return sum(fn_num(dset) for dset in dataset_name)
        else:
            return None

    def setup(self, stage):
        # self.use_ddp = self.trainer.use_ddp
        if not hasattr(self, "use_ddp"): self.use_ddp = self.trainer.use_ddp
        if stage == 'fit':
            if self.use_ddp:
                num_proc = self.trainer.num_nodes * self.trainer.num_processes
                self.hparams.lr *= num_proc
                self.logger.experiment.info(f"world-size: {num_proc}")

        self.val_len = 1 if not isinstance(
            self.hparams.val_dataset, list) else len(self.hparams.val_dataset)

        self.acc_meter = [
            utils_system.AverageMeter(use_ddp=self.use_ddp)
            for _ in range(self.val_len)
        ]
        self.best_metric = [[torch.tensor(-1.), -1]
                            for _ in range(self.val_len)]
        # self.aggr_class = [AggregatePred() for _ in range(self.val_len)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        if parent_parser is None:
            parent_parser = []
        else:
            parent_parser = [parent_parser]
        parser = argparse.ArgumentParser(parents=parent_parser,
                                         conflict_handler='resolve')
        parser.add_argument("--linear_bn_affine", type=str, default='false')
        return parser

    @abc.abstractmethod
    def forward(self, x):
        return

    def load_base(self, ckpt_path=None):
        if ckpt_path is not None:
            if self.hparams.model == 'moco':
                # load state dict from mocov2 pretrained weights
                state_dict = torch.load(
                    ckpt_path,
                    map_location=lambda storage, loc: storage)['state_dict']
                resnet_head = torchvision.models.resnet50()
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith(
                            'module.encoder_q.fc'):
                        # remove prefix
                        state_dict[
                            k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                resnet_head.load_state_dict(state_dict, strict=False)
                encoder = nn.Sequential(*(list(resnet_head.children())[:-1] +
                                          [nn.Flatten()]))
                self.feature.load_state_dict(encoder.state_dict())
            else:
                ckpt = torch.load(
                    ckpt_path,
                    map_location=lambda storage, loc: storage)['state_dict']
                new_state = {}
                for k, v in ckpt.items():
                    if 'feature.' in k:
                        new_state[k.replace('feature.', '')] = v
                self.feature.load_state_dict(
                    new_state, strict=not self.hparams.load_flexible)

    # --------------------------------- training --------------------------------- #

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        import time
        time.sleep(1.5)  # wait a bit to kill dataloader workers

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """ implement your own training step """

    # --------------------------------  validation -------------------------------- #
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if self.hparams.eval_mode == 'linear':
            return self._linear_validation_step(batch, batch_idx, dataset_idx)
        elif self.hparams.eval_mode == 'few_shot':
            return self.few_shot_validation_step(batch, batch_idx, dataset_idx)
        else:
            raise NotImplementedError

    def validation_epoch_end(self, outputs):
        if self.hparams.eval_mode == 'linear':
            return self._linear_validation_epoch_end(outputs)
        elif self.hparams.eval_mode == 'few_shot':
            return self.few_shot_val_end(outputs)
        else:
            raise NotImplementedError

    # ----------------------------------- test ----------------------------------- #
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    # --------------------------------- load data -------------------------------- #
    def prepare_data(self):
        self.dm.prepare_data()

    def train_dataloader(self):
        if self.hparams.val_as_test == 'false':
            return self.dm.train_dataloader(pl_trainer=self.trainer,
                                            use_ddp=self.use_ddp)
        elif self.hparams.val_as_test == 'true':
            return self.dm.train_dataloader_val_split(pl_trainer=self.trainer,
                                                      use_ddp=self.use_ddp)

    def val_dataloader(self):
        if self.hparams.disable_validation:
            return None
        if self.hparams.eval_mode == 'few_shot':
            return self.dm.get_fewshot_dataloader(pl_trainer=self.trainer,
                                                  use_ddp=self.use_ddp)
        elif self.hparams.eval_mode == 'linear':
            if self.hparams.val_as_test == 'true':
                return self.dm.val_dataloader_val_split(
                    pl_trainer=self.trainer, use_ddp=self.use_ddp)

            base_loader = self.dm.get_simple_dataloader(
                self.hparams.val_dataset,
                aug=self.hparams.val_aug,
                pl_trainer=self.trainer,
                use_ddp=self.use_ddp,
                opt=self.hparams,
                shuffle=self.hparams.shuffle_val)
            return base_loader

    def test_dataloader(self):
        return self.val_dataloader()

    # ---------------------------------- config ---------------------------------- #
    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                               self.parameters()),
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.optim_momentum,
                                        weight_decay=self.hparams.optim_wd)
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                self.parameters()),
                                         lr=self.hparams.lr,
                                         weight_decay=self.hparams.optim_wd)
        elif self.hparams.optimizer == "RMSProp":
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad,
                                                   self.parameters()),
                                            lr=self.hparams.lr,
                                            weight_decay=self.hparams.optim_wd)
        elif self.hparams.optimizer == "LARS":
            from utils.lars import LARSWrapper
            optimizer = LARSWrapper(
                torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       self.parameters()),
                                lr=self.hparams.lr,
                                momentum=self.hparams.optim_momentum,
                                weight_decay=self.hparams.optim_wd))
        else:
            raise ValueError("provide a supported optimizer")
        return optimizer

    # def on_train_epoch_start(self) -> None:
    #     super().on_train_epoch_start()
    #     self.log_lr()

    def log_lr(self):
        # log learning rate
        lr = next(iter(self.trainer.optimizers[0].param_groups))['lr']
        self.log("lr",
                 lr,
                 on_epoch=True,
                 on_step=False,
                 logger=True,
                 prog_bar=False)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, *args, **kwargs):
        if optimizer_idx == 0:
            # warm up lr
            if self.hparams.warm_up == 'true':
                if epoch < self.hparams.warm_epochs:
                    self.lr_warmp_up(optimizer)
                else:
                    self.adjust_learning_rate(optimizer,
                                              epoch - self.hparams.warm_epochs)

            else:
                self.adjust_learning_rate(optimizer, epoch)

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def lr_warmp_up(self, optimizer):
        warmup_from = self.hparams.lr * 1e-2
        warmup_to = self.hparams.lr
        total_warm_up_step = self.trainer.num_training_batches * self.hparams.warm_epochs
        lr_scale = min(
            1.,
            float(self.trainer.global_step + 1) / total_warm_up_step)
        lr = warmup_from + lr_scale * (warmup_to - warmup_from)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch < 0:
            return
        base_lr = self.hparams.lr
        if self.hparams.scheduler == 'cosine':
            eta_min = 1e-8
            lr = eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.hparams.max_epochs)) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif self.hparams.scheduler == 'step':
            if not hasattr(self, '_milestones'):
                self._milestones = {
                    int(_v): i + 1
                    for i, _v in enumerate(
                        self.hparams.step_lr_milestones.split(","))
                }
            if epoch in self._milestones:
                _lr = (base_lr *
                       self.hparams.lr_decay_rate**self._milestones[epoch])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = _lr
        elif self.hparams.scheduler == 'exponential':
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (
                    1 - self.hparams.lr_decay_rate)
        else:
            return

    # ----------------------------------- model ---------------------------------- #
    def _create_model(self, num_class):
        """ create encoder and classifier head """

        self.feature = self.base_encoder()
        if num_class is None:
            self.classifier = None
            return

        if self.hparams.pretrained and num_class == 1000:
            fc = getattr(
                torchvision.models,
                self.hparams.model)(pretrained=self.hparams.pretrained).fc
            self.classifier = fc
            return

        # classifier head
        if not self.hparams.linear_bn:
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        else:
            # add a batchnorm before classifier
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.feature.final_feat_dim,
                               affine=self.hparams.linear_bn_affine == 'true'),
                nn.Linear(self.feature.final_feat_dim, num_class))

    def base_encoder(self):
        if self.hparams.system_mode == "2d":
            if self.hparams.model.lower() == "resnet10":
                from utils import backbone
                encoder = backbone.ResNet10(flatten=True)
                # raise NotImplementedError
            elif self.hparams.model.lower() == "resnet12":
                from utils.resnet12_backbone import resnet12
                encoder = resnet12(avg_pool=True)
                encoder.final_feat_dim = 640
            elif "vgg" in self.hparams.model:
                mod = getattr(torchvision.models, self.hparams.model)(
                    pretrained=self.hparams.pretrained).features
                encoder = nn.Sequential(mod, nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten())
                encoder.final_feat_dim = 512
            elif "random" in self.hparams.model:
                encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
                encoder.final_feat_dim = 5
            elif "resnet" in self.hparams.model:
                resnet_head = getattr(
                    resnet_models,
                    self.hparams.model)(pretrained=self.hparams.pretrained)

                encoder = nn.Sequential(*(list(resnet_head.children())[:-1] +
                                          [nn.Flatten()]))
                encoder.final_feat_dim = resnet_head.fc.in_features
            elif "moco" in self.hparams.model:
                resnet_head = getattr(
                    resnet_models,
                    'resnet50')(pretrained=self.hparams.pretrained)

                encoder = nn.Sequential(*(list(resnet_head.children())[:-1] +
                                          [nn.Flatten()]))
                # self.resnet_head = resnet_head
                encoder.final_feat_dim = resnet_head.fc.in_features
            else:
                raise NotImplementedError(
                    f"{self.hparams.model} not implemented")
        else:
            mod = torchvision.models.video.__dict__[self.hparams.model](
                pretrained=self.hparams.pretrained)
            encoder = nn.Sequential(*list(mod.children())[:-1], nn.Flatten())
            encoder.final_feat_dim = 512
        return encoder

    def get_feature_extractor(self):
        return self.feature

    def ece_calculate(self, prob, gt):
        return calc_ece(prob, gt)

    # --------------------------------- finetune --------------------------------- #


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.fc(x)
        return x


class AggregatePred:
    def __init__(self) -> None:
        self.reset()

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               label_seq: torch.Tensor = None,
               concat_all=False) -> None:
        if concat_all:
            pred = concat_all_ddp(pred)
            gt = concat_all_ddp(gt)

        pred_np = pred.data.cpu().numpy().tolist()
        gt_np = gt.data.cpu().numpy().tolist()
        if label_seq is not None:
            label_map = self.get_label_map(label_seq)
            pred_np = [label_map[x] for x in pred_np]
            gt_np = [label_map[x] for x in gt_np]
        self.gt.extend(gt_np)
        self.pred.extend(pred_np)

        # return current accuracy
        return accuracy_score(gt_np, pred_np)

    def get_label_map(self, label_seq: torch.Tensor):
        return {i: c.item() for i, c in enumerate(label_seq)}

    def get_report(self):
        return classification_report(self.gt, self.pred)

    def get_accuracy(self):
        return accuracy_score(self.gt, self.pred)

    def reset(self) -> None:
        self.gt = []
        self.pred = []


@torch.no_grad()
def concat_all_ddp(tensor):
    try:
        world_size = self.trainer.num_nodes * self.trainer.num_processes
    except AssertionError:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output