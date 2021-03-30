import torch
import torch.nn as nn
import numpy as np
import torch
import copy
import numpy as np
import system.utils_finetune as ftune
from pytorch_lightning.metrics.functional import accuracy


class FewShotMixin():
    def few_shot_validation_step(self, batch, batch_idx, dataset_idx=0):
        out = self.few_shot_finetune(batch, dataset_idx)
        if self.hparams.print_val:
            self.logger.log(
                f"val :: ({self.val_names[dataset_idx]}) : ({batch_idx}) : {out['acc']:.4f}, acc_mean: {out['acc_avg']:.4f}",
                level="DEBUG")

    def few_shot_val_end(self, outputs):
        mean_val = []
        # max_val = []
        for dataset_idx in range(self.val_len):
            acc_mean = self.acc_meter[dataset_idx].mean
            acc_std = self.acc_meter[dataset_idx].std

            if self.logger is not None:
                self.logger.log(
                    f"Test Acc ({self.val_names[dataset_idx]}) = {acc_mean:4.4f} +- {acc_std:4.4f}",
                    level="DEBUG")

            self.acc_meter[dataset_idx].reset()

            mean_val.append(acc_mean)

        acc_mean = torch.mean(torch.stack(mean_val))
        tqdm_dict = {
            "acc_mean": acc_mean,
        }

        self.log_dict(tqdm_dict, prog_bar=True)

    def few_shot_finetune(self, batch, dataset_idx=0):
        # NOTE: grad is enabled
        x, y = batch
        n_way = self.hparams.test_n_way
        n_support = self.hparams.n_shot

        n_query = x.size(1) - n_support
        x_var = x

        y_a_i = torch.from_numpy(np.repeat(range(n_way),
                                           n_support)).to(self.device)  # (25,)

        x_b_i = x_var[:, n_support:, :, :, :].contiguous().view(
            n_way * n_query,
            *x.size()[2:])
        x_a_i = x_var[:, :n_support, :, :, :].contiguous().view(
            n_way * n_support,
            *x.size()[2:])  # (25, 3, 224, 224)

        y_query = torch.arange(n_way, device=self.device,
                               dtype=x.dtype).repeat_interleave(n_query)

        encoder = self.get_feature_extractor()

        if self.hparams.fine_tune_type == "deep":
            pretrained_model = copy.deepcopy(encoder)
            classifier = Classifier(pretrained_model.final_feat_dim, n_way)
            classifier.to(self.device)
            classifier.train()

            scores = ftune.finetune_deep(
                pretrained_model,
                classifier,
                x_a_i,
                y_a_i,
                x_b_i,
                y_test=None,
                freeze_backbone=self.hparams.freeze_backbone,
                total_epoch=self.hparams.deep_finetune_epoch,
                batch_size=self.hparams.deep_finetune_batch_size,
                use_norm=self.hparams.ft_normalize == "true",
                device=self.device,
                opt=self.hparams,
                last_k=self.hparams.deep_finetune_lastk)

            torch.set_grad_enabled(False)
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.squeeze()

            # free gpu spaces
            del pretrained_model
            del classifier

        elif self.hparams.fine_tune_type == "NN":
            topk_ind = ftune.NN(encoder,
                                x_a_i,
                                y_a_i,
                                x_b_i,
                                norm=self.hparams.ft_normalize == "true")

        elif self.hparams.fine_tune_type == "Cosine":
            topk_ind = ftune.Cosine(encoder,
                                    x_a_i,
                                    y_a_i,
                                    x_b_i,
                                    norm=self.hparams.ft_normalize == "true",
                                    n_way=n_way,
                                    n_support=n_support)
        elif self.hparams.fine_tune_type == "LR":
            topk_ind, _ = ftune.LR(encoder,
                                   x_a_i,
                                   y_a_i,
                                   x_b_i,
                                   norm=self.hparams.ft_normalize == "true")
        else:
            raise NotImplementedError
        _val = accuracy(topk_ind, y_query)
        self.acc_meter[dataset_idx].add(_val, n=1)

        return {
            "acc": self.acc_meter[dataset_idx].value,
            "acc_avg": self.acc_meter[dataset_idx].mean
        }


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.fc(x)
        return x
