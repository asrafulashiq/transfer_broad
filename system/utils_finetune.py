import torch
import numpy as np
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os
import copy
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pytorch_lightning.metrics.functional import accuracy
import kornia
import kornia.augmentation as Kaug


class ApplyAug(nn.Module):
    def __init__(self, im_size=224, device=torch.device('cuda:0')):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.aug = torch.nn.Sequential(
            kornia.geometry.transform.Resize(int(im_size * 1.2)),
            Kaug.RandomCrop((im_size, im_size), padding=8),
            Kaug.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            Kaug.RandomHorizontalFlip(),
            Kaug.Normalize(mean=self.mean, std=self.std))

    def forward(self, x):
        x = x * self.std[:, None, None] + self.mean[:, None, None]
        x = self.aug(x)
        return x


def finetune_proto(pretrained_model,
                   classifier,
                   x_train,
                   y_train,
                   x_test,
                   y_test=None,
                   freeze_backbone='true',
                   total_epoch=50,
                   batch_size=4,
                   use_norm=False,
                   n_way=5,
                   n_support=5,
                   device=torch.device('cuda:0'),
                   verbose=False,
                   distance_type="l2"):
    torch.set_grad_enabled(True)

    if classifier is None:
        classifier = nn.Identity()
    classifier.alpha = nn.Parameter(torch.tensor(1., device=device))
    # classifier.alpha = 1.

    if len(list(classifier.parameters())) > 0:
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr=0.01)
    #  momentum=0.9,
    #  dampening=0.9,
    #  weight_decay=0.001)
    else:
        classifier_opt = None

    if freeze_backbone == 'false':
        pretrained_model.train()
        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            pretrained_model.parameters()),
                                     lr=0.01)

    else:
        pretrained_model.eval()
        for par in pretrained_model.parameters():
            par.requires_grad = False
    support_size = x_train.shape[0]

    train_support_size = int(n_support * 0.7)
    val_support_size = n_support - train_support_size

    _, c, h, w = x_train.shape

    loss_fn = nn.CrossEntropyLoss()

    pretrained_model_comb = nn.Sequential(
        pretrained_model,
        Normalize() if use_norm else nn.Identity(), classifier)

    aug_cls = nn.Identity()  #ApplyAug(im_size=h)

    for epoch in range(total_epoch):
        rand_id = np.random.permutation(support_size)

        rand_support_ind = np.hstack([
            (np.random.choice(n_support, train_support_size, replace=False) +
             i * n_way) for i in range(n_way)
        ])
        if classifier_opt is not None:
            classifier_opt.zero_grad()
        if freeze_backbone == 'false':
            delta_opt.zero_grad()

        train_mask = np.zeros(support_size, dtype=bool)
        train_mask[rand_support_ind] = True

        x_embed = pretrained_model_comb(aug_cls(x_train))

        # n_way, c
        x_train_proto = x_embed[train_mask].view(n_way, train_support_size,
                                                 -1).mean(dim=1)
        x_val_s = x_embed[~train_mask]  # b, c

        # -1, n_way
        if distance_type == "l2":
            distance = (x_val_s[:, None] - x_train_proto[None]).pow(2).sum(
                dim=-1).neg().mul(classifier.alpha)
        else:
            distance = normalize(x_val_s) @ normalize(x_train_proto).transpose(
                0, 1)
            distance = distance.mul(classifier.alpha)
        # dis_soft = dis_soft.reshape(n_way, -1)
        # y_train_s = y_train[train_mask]
        y_val_s = y_train[~train_mask]

        loss = loss_fn(distance, y_val_s)

        #####################################
        loss.backward()
        if classifier_opt is not None:
            classifier_opt.step()

        if freeze_backbone == 'false':
            delta_opt.step()

    pretrained_model_comb.eval()
    x_embed = pretrained_model_comb(x_train)

    x_proto = x_embed.view(n_way, n_support, -1).mean(dim=1)
    with torch.no_grad():
        output = pretrained_model_comb(x_test)
        if distance_type == "l2":
            distance = (output[:, None] - x_proto[None]).pow(2).sum(
                dim=-1).neg().mul(classifier.alpha)
        else:
            distance = normalize(output) @ normalize(x_proto).transpose(0, 1)
            distance = distance.mul(classifier.alpha)
        scores = distance.softmax(dim=-1)

    return scores


def finetune_deep(pretrained_model,
                  classifier,
                  x_train,
                  y_train,
                  x_test,
                  y_test=None,
                  freeze_backbone='true',
                  total_epoch=50,
                  batch_size=4,
                  use_norm=False,
                  device=torch.device('cuda:0'),
                  verbose=False,
                  opt=None,
                  last_k=-1):
    torch.set_grad_enabled(True)
    loss_fn = nn.CrossEntropyLoss()
    classifier_opt = torch.optim.SGD(
        classifier.parameters(),
        lr=opt.deep_finetune_lr_class,
        momentum=opt.deep_finetune_momentum,
        #  dampening=0.9,
        weight_decay=opt.deep_finetune_wd)

    if freeze_backbone == 'false':
        pretrained_model.train()
        # do not finetune batchnorm layer
        if opt.deep_finetune_freeze_bn == 'true':
            set_bn_to_eval(pretrained_model)
        param_to_train = []
        if last_k == -1:
            param_to_train = filter(lambda p: p.requires_grad,
                                    pretrained_model.parameters())
        else:
            param_to_train = filter(
                lambda p: p.requires_grad,
                pretrained_model[-2 - last_k:-2].parameters())
        delta_opt = torch.optim.SGD(param_to_train,
                                    lr=opt.deep_finetune_lr_mod)
    else:
        pretrained_model.eval()
        for par in pretrained_model.parameters():
            par.requires_grad = False

    support_size = x_train.shape[0]
    for epoch in tqdm(range(total_epoch),
                      desc="Inner epoch",
                      position=4,
                      ncols=100,
                      disable=True):
        # rand_id = np.random.permutation(support_size)
        rand_id = torch.randperm(support_size)

        for j in range(0, support_size, batch_size):
            classifier_opt.zero_grad()
            if freeze_backbone == 'false':
                delta_opt.zero_grad()

            #####################################
            if j + batch_size > support_size:
                continue

            selected_id = rand_id[j:min(j + batch_size, support_size)]
            # selected_id = torch.from_numpy(
            #     rand_id[j:min(j + batch_size, support_size)]).to(device)

            z_batch = x_train[selected_id]
            y_batch = y_train[selected_id]
            #####################################

            output = pretrained_model(z_batch)
            if use_norm:
                output = normalize(output)
            output = classifier(output)
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()
            classifier_opt.step()

            if freeze_backbone == 'false':
                delta_opt.step()

        if verbose and epoch % 10 == 0:
            # train accuracy
            output = pretrained_model(x_train)
            if use_norm: output = normalize(output)
            output = classifier(output)
            scores = torch.argmax(output, dim=-1)
            train_accur = accuracy(scores, y_train)
            train_loss = loss_fn(output, y_train)

            logger.debug(
                f"epoch {epoch:>05d} :: train_accuracy {train_accur:.4f} train_loss {train_loss: .4f}"
            )
    torch.set_grad_enabled(False)

    pretrained_model.eval()
    classifier.eval()

    with torch.no_grad():
        output = pretrained_model(x_test)
        if use_norm:
            output = normalize(output)
        scores = classifier(output)
    return scores


def _finetune(pretrained_model,
              classifier,
              x_train,
              y_train,
              x_test,
              y_test=None,
              freeze_backbone='true',
              total_epoch=50,
              batch_size=4,
              use_norm=True,
              device=torch.device('cuda:0'),
              verbose=False,
              opt=None,
              last_k=-1,
              lr=0.01,
              optim='SGD',
              deep_finetune_freeze_bn='false',
              neg_loss=False):
    torch.set_grad_enabled(True)
    classifier.train()

    if optim == 'SGD':
        classifier_opt = torch.optim.SGD(classifier.parameters(),
                                         lr=lr,
                                         momentum=0.9)
    else:
        classifier_opt = torch.optim.Adam(classifier.parameters(),
                                          lr=lr,
                                          weight_decay=1e-5)

    if freeze_backbone == 'false':
        pretrained_model.train()
        # do not finetune batchnorm layer
        if deep_finetune_freeze_bn == 'true':
            set_bn_to_eval(pretrained_model)
        param_to_train = []
        if last_k == -1:
            for p in pretrained_model.parameters():
                p.requires_grad = True
            param_to_train = filter(lambda p: p.requires_grad,
                                    pretrained_model.parameters())
        else:
            for p in pretrained_model[:-2 - last_k].parameters():
                p.requires_grad = False
            for p in pretrained_model[-2 - last_k:-2].parameters():
                p.requires_grad = True
            param_to_train = filter(
                lambda p: p.requires_grad,
                pretrained_model[-2 - last_k:-2].parameters())
        if optim == 'SGD':
            delta_opt = torch.optim.SGD(param_to_train, lr=lr)
        else:
            delta_opt = torch.optim.Adam(param_to_train,
                                         lr=lr,
                                         weight_decay=1e-5)

    else:
        pretrained_model.eval()
        for par in pretrained_model.parameters():
            par.requires_grad = False

    support_size = x_train.shape[0]
    num_class = classifier.fc.weight.shape[0]

    for epoch in tqdm(range(total_epoch),
                      desc="Inner epoch",
                      position=4,
                      ncols=100,
                      disable=True):
        rand_id = np.random.permutation(support_size)

        list_loss = []

        for j in range(0, support_size, batch_size):
            classifier_opt.zero_grad()
            if freeze_backbone == 'false':
                delta_opt.zero_grad()

            #####################################
            if j + batch_size > support_size:
                continue

            selected_id = torch.from_numpy(
                rand_id[j:min(j + batch_size, support_size)]).to(device)

            z_batch = x_train[selected_id]
            y_batch = y_train[selected_id]
            #####################################

            output = pretrained_model(z_batch)
            if use_norm:
                output = normalize(output)
            output = classifier(output)
            if neg_loss is False:
                loss = F.cross_entropy(output, y_batch)
                _loss_train = loss
            else:
                y_mask_neg = torch.ones_like(output).scatter(
                    1, y_batch.unsqueeze(-1), 0)  # b, cls
                neg_ind = y_mask_neg.nonzero().reshape(y_batch.shape[0],
                                                       num_class - 1,
                                                       -1)[..., 1]  # b, cls-1
                _rand = torch.randint_like(y_batch[:, None], 0, num_class - 1)
                sel = neg_ind.gather(1, _rand)
                out_soft = output.softmax(dim=-1)
                loss = (1 - out_soft.gather(1, sel)).log().neg().mean()

                _loss_train = F.cross_entropy(output, y_batch)
                # out_soft = output.softmax(dim=-1)
                # loss = (-y_mask_neg *
                #         torch.log(1 - out_soft + 1e-6)).sum(dim=-1).mean()

            #####################################
            loss.backward()
            classifier_opt.step()

            list_loss.append(_loss_train.item())

            if freeze_backbone == 'false':
                delta_opt.step()

        if verbose and epoch % 10 == 0:
            with torch.no_grad():
                # train accuracy
                output = pretrained_model(x_test)
                if use_norm: output = normalize(output)
                output = classifier(output)
                scores = torch.argmax(output, dim=-1)
                _accur = accuracy(scores, y_test)
                _loss = F.cross_entropy(output, y_test)

            logger.debug(
                f"epoch {epoch:>05d} :: test_accuracy {_accur:.2f} test_loss {_loss: .2f}"
                + f" train_loss {np.mean(list_loss):.2f}")
    torch.set_grad_enabled(False)

    pretrained_model.eval()
    classifier.eval()

    with torch.no_grad():
        output = pretrained_model(x_test)
        if use_norm:
            output = normalize(output)
        scores = classifier(output)
    return scores


def episodic_train_test(pretrained_model,
                        classifier,
                        unlbl_dataset,
                        x_train,
                        y_train,
                        y_train_actual,
                        x_test,
                        y_test=None,
                        freeze_backbone='true',
                        total_epoch=50,
                        batch_size=12,
                        use_norm=False,
                        device=torch.device('cuda:0'),
                        verbose=False,
                        opt=None,
                        last_k=-1,
                        thresh=0.6):

    y_test = y_test.long()
    scores = _finetune(pretrained_model,
                       classifier,
                       x_train,
                       y_train,
                       x_test,
                       y_test,
                       True,
                       total_epoch,
                       batch_size,
                       use_norm,
                       device,
                       verbose,
                       opt,
                       last_k=-1,
                       lr=opt.deep_finetune_lr_class,
                       optim='SGD',
                       deep_finetune_freeze_bn=opt.deep_finetune_freeze_bn)
    scores = scores.argmax(dim=-1)
    accuracy_before = accuracy(scores, y_test)
    print(f"accuracy before: {accuracy_before.item()*100:.2f}")

    with torch.no_grad():
        support = pretrained_model(x_train)
        support_proto = support.reshape(opt.test_n_way, opt.n_shot,
                                        -1).mean(dim=-2)
        support_proto = F.normalize(support_proto, dim=-1)
        support_ys = y_train[::opt.n_shot]

    # get pseudo-label from loader
    pseudo_loader = torch.utils.data.DataLoader(unlbl_dataset,
                                                batch_size=500,
                                                shuffle=True,
                                                num_workers=12,
                                                drop_last=False)
    all_gt = []
    all_pred = []
    label_map = {a.item(): b.item() for a, b in zip(y_train, y_train_actual)}

    X_n = []
    Y_n = []
    Y_n_tmp = []
    Y_in_domain = []

    _label_map = {b.item(): a.item() for a, b in zip(y_train, y_train_actual)}

    for i, batch in enumerate(pseudo_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = pretrained_model(x)
            if use_norm:
                output = normalize(output)

            # output = F.normalize(output, dim=-1)
            # pred = output @ support_proto.transpose(-1, -2)
            # p_max, p_max_ind = pred.max(dim=-1)
            # X_n.append(x[p_max > thresh])
            # Y_n.append(p_max_ind[p_max > thresh])
            # Y_n_tmp.extend(
            #     [_label_map.get(p.item(), -1) for p in y[p_max > thresh]])

            pred = classifier(output)

            p_max, p_max_ind = pred.max(dim=-1)
            X_n.append(x[p_max > thresh])
            Y_n.append(p_max_ind[p_max > thresh])
            Y_n_tmp.extend(
                [_label_map.get(p.item(), -1) for p in y[p_max > thresh]])

            # Y_n_tmp.append(p_max_ind[p_max > thresh])

            in_domain = torch.eq(y.unsqueeze(-1),
                                 y_train_actual.unsqueeze(-1).T).sum(-1) > 0

            Y_in_domain.append(in_domain[p_max > thresh])

            # _l = (p_max > thresh).nonzero()[:, 0]
            # if len(_l) > 0:
            #     all_gt.extend(y[_l].data.cpu().numpy().tolist())
            #     all_pred.extend(
            #         [label_map[p.item()] for p in pred[_l].argmax(-1)])

            _l = (pred.max(dim=-1)[0] > thresh).nonzero()[:, 0]
            if len(_l) > 0:
                all_gt.extend(y[_l].data.cpu().numpy().tolist())
                all_pred.extend(
                    [label_map[p.item()] for p in pred[_l].argmax(-1)])
        if i > 5:
            break

    report = confusion_matrix(all_gt, all_pred, normalize='pred')
    # tmp_ind = np.where(report.sum(0) == 0)[0]
    tmp_ind = []

    print("    " + " ".join(
        [f"{tmp:>2d}" for tmp in range(len(report)) if tmp not in tmp_ind]))

    for ir, r in enumerate(report):
        if ir in tmp_ind:
            continue
        print(f"{ir:>2d}:", end=" ")
        for ic, c in enumerate(r):
            if ic not in tmp_ind:
                print(f"{int(c*100):>02d}", end=" ")
        print()
    pass

    # raise SystemExit

    X_n = torch.cat(X_n, dim=0)
    Y_n = torch.cat(Y_n, dim=0)

    Y_n_tmp = torch.tensor(Y_n_tmp, dtype=torch.long).to(X_n.device)

    Y_in_domain = torch.cat(Y_in_domain, dim=0)

    X_n = X_n[Y_in_domain]
    Y_n = Y_n[Y_in_domain]
    Y_n_tmp = Y_n_tmp[Y_in_domain]
    print("accur:", accuracy(Y_n, Y_n_tmp))

    x_new = torch.cat((x_train, X_n), dim=0)
    y_new = torch.cat((y_train, Y_n), dim=0)
    scores = _finetune(
        pretrained_model,
        classifier,
        x_new,
        y_new,
        x_test,
        y_test,
        freeze_backbone=freeze_backbone,
        total_epoch=100,
        batch_size=min(20, X_n.shape[0] // 4),
        use_norm=use_norm,
        device=device,
        verbose=True,
        opt=opt,
        last_k=1,
        lr=0.1,  #5e-3,
        optim='SGD',
        deep_finetune_freeze_bn=opt.deep_finetune_freeze_bn,
        neg_loss=True)
    return scores


def NN(model, support, support_ys, query, norm=False, n_way=5, n_support=5):
    """nearest classifier"""
    support = model(support)
    query = model(query)

    # average support features
    support = support.reshape(n_way, n_support, -1).mean(dim=-2)
    support_ys = support_ys[::n_support]

    if norm:
        support = normalize(support)
        query = normalize(query)
    distance = torch.norm(support[None] - query[:, None], dim=-1, p=2)
    min_idx = torch.argmin(distance, dim=-1).squeeze()
    pred = support_ys[min_idx]
    return pred


@torch.no_grad()
def LR(model, support, support_ys, query, norm=False):
    """logistic regression classifier"""

    # augmenter = ApplyAug()

    # aug_sup = [support]
    # for i in range(4):
    #     aug_sup.append(augmenter(support))

    # support = torch.cat(aug_sup, dim=0)
    # support_ys = support_ys.repeat(5)

    support = model(support).detach()
    query = model(query).detach()
    if norm:
        support = normalize(support)
        query = normalize(query)

    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    support_features_np = support.data.cpu().numpy()
    support_ys_np = support_ys.data.cpu().numpy()
    clf.fit(support_features_np, support_ys_np)

    query_features_np = query.data.cpu().numpy()
    query_ys_pred = clf.predict(query_features_np)

    pred = torch.from_numpy(query_ys_pred).to(support.device)
    return pred, query


def Cosine(model, support, support_ys, query, norm=False, n_way=5,
           n_support=5):
    """Cosine classifier"""

    support = model(support).detach()
    query = model(query).detach()

    # average support features
    support = support.reshape(n_way, n_support, -1).mean(dim=-2)
    support_ys = support_ys[::n_support]

    support_norm = support / (support.norm(dim=-1, keepdim=True) + 1e-6)
    query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-6)
    cosine_distance = query_norm @ support_norm.transpose(-1, -2)

    max_idx = torch.argmax(cosine_distance, dim=-1).squeeze()

    pred = support_ys[max_idx]

    return pred


def normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=-1):
        return normalize(x, dim)


def set_bn_to_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False