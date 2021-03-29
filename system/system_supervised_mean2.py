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
        parser.add_argument("--model_name",
                            type=str,
                            default="supervised_mean2")

        parser.add_argument("--moco_K", type=int, default=64000)
        parser.add_argument("--moco_T1", type=float, default=0.07)
        parser.add_argument("--moco_T2", type=float, default=0.07)

        parser.add_argument("--lm_intra", type=float, default=0.5)
        parser.add_argument("--lm_inter", type=float, default=1)
        parser.add_argument("--proto",
                            default='true',
                            choices=['true', 'false'])

        parser.add_argument("--intra_all",
                            default='true',
                            choices=['true', 'false'])
        parser.add_argument("--moco_m_cluster", type=float, default=0.90)
        parser.add_argument("--moco_num_cluster", type=int, default=1)

        parser.add_argument("--moco_eqco", type=float, default=None)
        parser.add_argument("--mlp_feature_branch", type=str, default="both")
        return parser

    # --------------------------------- training --------------------------------- #
    def _training_step(self, batch, batch_idx):
        data, label = batch
        xq, xk = data[0], data[1]

        logits, mask, logits_same, labels_same, loss_diff, loss_same = self.moco.forward(
            xq, xk, label)

        with torch.no_grad():
            acc1, _ = moco_accuracy(logits_same, labels_same, topk=(1, 5))

        loss = (self.hparams.lm_intra * loss_same +
                self.hparams.lm_inter * loss_diff)

        tqdm_dict = {
            "loss_train": loss,
            "loss_intra": loss_same,
            "loss_inter": loss_diff,
            "top1.val": acc1,
            # "top1_diff.val": self.top1_diff.val,
        }

        self.log_dict(tqdm_dict,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True)
        return loss

    # ----------------------------------- model ---------------------------------- #
    def _create_moco(self, enc_q, enc_k):
        self.moco = MoCo(enc_q,
                         enc_k,
                         dim=enc_q.final_feat_dim,
                         dim_hidden=self.hparams.moco_dim_hidden,
                         dim_mlp=self.hparams.moco_dim_final,
                         K_total=self.hparams.moco_K,
                         T1=self.hparams.moco_T1,
                         T2=self.hparams.moco_T2,
                         mlp=self.hparams.moco_mlp,
                         m_cluster=self.hparams.moco_m_cluster,
                         num_cluster=self.hparams.moco_num_cluster,
                         num_classes=self.num_classes,
                         use_ddp=self.use_ddp,
                         intra_all=self.hparams.intra_all,
                         m_eqco=self.hparams.moco_eqco,
                         proto=self.hparams.proto)

    def get_feature_extractor(self):
        """ return feature extractor """
        if self.hparams.moco_mlp_to_feature == 0:
            return self.feature
        else:
            return Feature_extractor(self.feature,
                                     self.moco.encoder_q[-1],
                                     opt=self.hparams)


class Feature_extractor(nn.Module):
    def __init__(self, backbone, header, opt=None):
        super().__init__()
        self.backbone = backbone
        self.header = header
        self.opt = opt

    def forward(self, x):
        x_feat = self.backbone(x)
        if self.opt.moco_mlp_to_feature == 0:
            return x_feat
        if self.opt.mlp_feature_branch == 'intra':
            extra = self.header.lin1[:self.opt.moco_mlp_to_feature]
            out = extra(x_feat)
        elif self.opt.mlp_feature_branch == 'inter':
            extra = self.header.lin2[:self.opt.moco_mlp_to_feature]
            out = extra(x_feat)
        else:
            extra1 = self.header.lin1[:self.opt.moco_mlp_to_feature]
            extra2 = self.header.lin2[:self.opt.moco_mlp_to_feature]
            out1 = extra1(x_feat)
            out2 = extra2(x_feat)
            out = torch.cat((out1, out2), dim=-1)
        return out


class Classifier_d(nn.Module):
    def __init__(self, dim, dim_hidden, n_way, drop_p=0):
        super().__init__()
        self.n_way = n_way
        if dim_hidden is None:
            dim_hidden = dim
        self.dropout = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
        self.lin1 = nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU(),
                                  nn.Linear(dim_hidden, n_way))
        self.lin2 = nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU(),
                                  nn.Linear(dim_hidden, n_way))

    def forward(self, x):
        x = self.dropout(x)
        x1 = F.normalize(self.lin1(x), dim=-1)
        x2 = F.normalize(self.lin2(x), dim=-1)
        return torch.cat((x1, x2), dim=-1)


# ---------------------------------------------------------------------------- #
#                                  MOCO Model                                  #
# ---------------------------------------------------------------------------- #
class MoCo(nn.Module):
    def __init__(self,
                 encoder_q,
                 encoder_k,
                 dim=2048,
                 dim_hidden=2048,
                 dim_mlp=128,
                 K_total=None,
                 m=0.999,
                 m_cluster=0.99,
                 num_cluster=1,
                 T1=0.5,
                 T2=1,
                 mlp=True,
                 num_classes=2,
                 use_ddp=False,
                 intra_all='false',
                 m_eqco=None,
                 proto=True):
        super(MoCo, self).__init__()

        self._use_ddp = None
        self.m = m

        self.m_cluster = m_cluster
        self.num_cluster = num_cluster

        self.T1 = T1
        self.T2 = T2
        self.num_classes = num_classes
        self.dim_mlp = dim_mlp
        self.proto = proto
        self.m_eqco = m_eqco

        self.intra_all = intra_all

        self.K = K_total

        if mlp == 'true':
            self.encoder_q = nn.Sequential(
                encoder_q, Classifier_d(dim, dim_hidden, dim_mlp))
            self.encoder_k = nn.Sequential(
                encoder_k, Classifier_d(dim, dim_hidden, dim_mlp))
        else:
            self.encoder_q = encoder_q
            self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(2 * dim_mlp, self.K))
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
        q = self.encoder_q(im_q)  # queries: NxC
        # NOTE: normalize already performed inside model
        # q = nn.functional.normalize(q, dim=1)

        q1, q2 = q[..., :self.dim_mlp], q[..., self.dim_mlp:]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.use_ddp:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            if self.use_ddp:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k1, k2 = k[..., :self.dim_mlp], k[..., self.dim_mlp:]

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)  # -> (n, 1)
        l_pos_q = (q1 @ self.queue[:self.dim_mlp].clone().detach())

        _lsame = torch.cat([l_pos, l_pos_q], dim=-1)
        logits_same = _lsame.div(self.T1)  # -> (n, 1+k')

        queue = self.queue.clone().detach()

        l_pos = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)  # -> (n, 1)
        l_neg = torch.einsum('nc,ck->nk', [q1, queue[:self.dim_mlp]])

        mask = torch.eq(labels[:, None], self.queue_index[:, None].T)  # nk
        if self.intra_all == 'false':
            if self.m_eqco is not None:
                l_pos = l_pos - self.m_eqco
            nmask = mask.clone().logical_not().float()
            nmask = nmask * torch.finfo().min
            l_neg = l_neg + nmask
        else:
            pass

        _lsame = torch.cat([l_pos, l_neg], dim=-1)
        logits_same = _lsame.div(self.T1)
        labels_same = torch.zeros(logits_same.shape[0],
                                  dtype=torch.long).to(k.device)
        loss_intra = F.cross_entropy(logits_same, labels_same)

        # dequeue and enqueue
        # mit-like loss
        self._dequeue_and_enqueue(k, labels)

        ### Supervised contrastive
        logits = torch.einsum(
            'nc,ck->nk',
            [q2, self.queue[self.dim_mlp:].clone().detach()]).div(self.T2)

        mask = torch.eq(labels[:, None], self.queue_index[:, None].T)  # nk
        # l_all_soft = l_all.softmax(dim=-1)
        loss_sup = (-torch.log_softmax(logits, dim=-1) * mask).sum(
            dim=-1, keepdim=True).div(mask.sum(dim=-1, keepdim=True) + 1e-5)
        loss_inter = loss_sup.mean()

        return logits, mask, logits_same, labels_same, loss_inter, loss_intra

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
        self.queue_index[_idx] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


def moco_accuracy_more(output, target, topk=(100, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        correct = target.gather(1, pred)

        res = []
        for k in topk:
            correct_k = correct[:, :k].sum().float() / k
            res.append(correct_k.mul(100.0 / batch_size))
        return res