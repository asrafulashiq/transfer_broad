from pytorch_lightning.metrics.functional import accuracy
import torch
import uncertainty_metrics.numpy as um
import torch.nn as nn
import torch.nn.functional as F

from .utils_system import concat_all_ddp, aggregate_dict


class LinearMixin():
    def _forward_batch(self, x):
        return self.forward(x)

    def _linear_validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        mlp_preds, *_ = self._forward_batch(x)
        mlp_loss = torch.nn.functional.cross_entropy(mlp_preds, y)
        self.log("linear_loss", mlp_loss, on_epoch=True)
        # result.loss = mlp_loss
        prob = mlp_preds.softmax(dim=-1)
        gt = y
        return {"prob": prob, "gt": gt}

    def _linear_validation_epoch_end(self, outputs, *args, **kwargs):
        outputs = aggregate_dict(outputs)
        epoch_probs = concat_all_ddp(outputs["prob"])
        epoch_gt = concat_all_ddp(outputs["gt"])
        epoch_preds = torch.argmax(epoch_probs, dim=-1)

        mean_accuracy = accuracy(epoch_preds, epoch_gt)

        ece_score = calc_ece(epoch_probs, epoch_gt)
        self.log("validation_ece", ece_score)

        self.log("acc_mean", mean_accuracy)


def calc_ece(probs, gt):
    ece_calc = _ECELoss()
    ece_score = ece_calc.forward(probs, gt)
    return ece_score


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels):
        # softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return ece