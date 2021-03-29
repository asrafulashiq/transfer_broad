from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

from system import system_abstract_moco
from .utils_system import concat_all_gather, moco_accuracy


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract_moco.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = system_abstract_moco.LightningSystem.add_model_specific_args(
            parent_parser)
        parser.add_argument("--model_name", type=str, default="bplusm")
        parser.add_argument("--lr", type=float, default=0.02)
        parser.add_argument("--batch_size", type=int, default=64)

        # moco
        parser.add_argument("--moco_K", type=int, default=64000)
        parser.add_argument("--moco_T", type=float, default=0.07)

        parser.add_argument("--eval_mode", type=str, default="linear")
        parser.add_argument("--intra_all",
                            default='true',
                            choices=['true', 'false'])
        parser.add_argument("--lm_moco", type=float, default=0.5)
        parser.add_argument("--build_linear_evaluator",
                            type=str,
                            default="false")

        return parser

    def forward(self, xq, xk, labels):
        output, target, out_enc = self.moco(xq, xk, labels)
        out_cls = self.base_head(out_enc)
        return output, target, out_cls

    # --------------------------------- training --------------------------------- #
    def _training_step(self, batch, batch_idx):
        data, label = batch
        xq, xk = data[0], data[1]

        output, target, out_cls = self(xq, xk, label)

        mlp_loss = F.cross_entropy(out_cls, label)
        moco_loss = F.cross_entropy(output, target)
        loss = mlp_loss + self.hparams.lm_moco * moco_loss

        acc1, _ = moco_accuracy(output, target, topk=(1, 5))
        preds = torch.max(out_cls, dim=-1)[1]
        try:
            acc = accuracy(preds, label)
        except RuntimeError:
            acc = torch.tensor(0., device=self.device, dtype=self.dtype)

        tqdm_dict = {
            "loss_train": loss,
            "moco_loss": moco_loss,
            "mlp_loss": mlp_loss,
            "top1.val": acc,
            "top1_moco": acc1
        }

        self.log_dict(tqdm_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _forward_batch(self, x):
        if len(x) == 2 and (isinstance(x, list) or isinstance(x, tuple)):
            x = x[0]
        with torch.no_grad():
            feature = self.feature(x)
        out = self.base_head(feature)
        return out, feature

    # ----------------------------------- model ---------------------------------- #
    def _create_moco(self, enc_q, enc_k):
        self.moco = MoCo(enc_q,
                         enc_k,
                         dim=enc_q.final_feat_dim,
                         dim_hidden=self.hparams.moco_dim_hidden,
                         dim_mlp=self.hparams.moco_dim_final,
                         K=self.hparams.moco_K,
                         T=self.hparams.moco_T,
                         mlp=self.hparams.moco_mlp,
                         intra_all=self.hparams.intra_all,
                         use_ddp=self.use_ddp)
        self.base_head = nn.Linear(enc_q.final_feat_dim, self.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           self.parameters()),
                                    lr=self.hparams.lr,
                                    momentum=self.hparams.optim_momentum,
                                    weight_decay=self.hparams.optim_wd)
        return optimizer


# ---------------------------------------------------------------------------- #
#                                  MOCO Model                                  #
# ---------------------------------------------------------------------------- #


class MoCo(nn.Module):
    def __init__(self,
                 encoder_q,
                 encoder_k,
                 dim=2048,
                 dim_hidden=None,
                 dim_mlp=128,
                 K=8192,
                 m=0.999,
                 intra_all='false',
                 T=0.07,
                 mlp='true',
                 use_ddp=False):
        super(MoCo, self).__init__()

        self._use_ddp = None

        self.K = K
        self.m = m
        self.T = T
        self.intra_all = intra_all

        if dim_hidden is None:
            dim_hidden = dim

        if mlp == 'true':  # hack: brute-force replacement
            self.encoder_q = nn.Sequential(encoder_q,
                                           nn.Linear(dim,
                                                     dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_mlp))
            self.encoder_k = nn.Sequential(encoder_k,
                                           nn.Linear(dim,
                                                     dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_mlp))
        else:
            self.encoder_q = encoder_q
            self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim_mlp, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "queue_index",
            torch.full((self.K, ), fill_value=-1, dtype=torch.long))

    @property
    def use_ddp(self):
        return self._use_ddp

    @use_ddp.setter
    def use_ddp(self, val):
        self._use_ddp = val

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        if self.use_ddp:
            keys = concat_all_gather(keys)
            labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        _idx = [i % self.K for i in range(ptr, ptr + batch_size)]
        self.queue[:, _idx] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_index[_idx] = labels

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]

        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, labels):
        assert isinstance(self.use_ddp, bool)
        # compute query features
        q_enc = self.encoder_q[0](im_q)  # queries: NxC
        q = nn.functional.normalize(self.encoder_q[1:](q_enc), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.use_ddp:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            if self.use_ddp:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        if self.intra_all == 'false':
            mask = torch.eq(labels[:, None], self.queue_index[:, None].T)  # nk
            nmask = mask.logical_not().float()
            nmask = nmask * torch.finfo().min
            l_neg = l_neg + nmask
        else:
            pass
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels_for_softmax = torch.zeros(logits.shape[0],
                                         dtype=torch.long).to(q.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return logits, labels_for_softmax, q_enc
