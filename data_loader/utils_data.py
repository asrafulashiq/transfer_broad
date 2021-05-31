import torch
import numpy as np
import torchvision.transforms as transforms
from abc import abstractmethod
from torchvision.datasets import ImageFolder
from PIL import ImageFile
import pytorch_lightning as pl
from tqdm import tqdm
from itertools import chain
from loguru import logger
from data_loader import additional_data, helper
from pytorch_lightning.utilities.distributed import rank_zero_only
from .additional_transforms import TwoCropsTransform, GaussianBlur
import os
from .additional_data import SimpleUnlabelDataset, ConcatProportionDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

identity = lambda x: x

# NOTE: temporary store cache in TMP_PATH
TMP_PATH = os.path.expanduser("~/.tmp")
DATA_ROOT = os.path.expanduser("~/datasets/cdfsl")

os.makedirs(TMP_PATH, exist_ok = True)

def get_split(dname):
    splits = dname.split("_")
    if len(splits) > 1:
        base, mode = splits[0], splits[-1]
    else:
        base, mode = splits[0], None
    # These datasets have no train/test split, manually create them
    # SUN397, ISIC, ChestX, EuroSAT, Omniglot, sketch, DeepWeeds, Resisc45
    dataset_no_split = [
        "SUN397", "ISIC", "ChestX", "EuroSAT", "Omniglot", "Sketch",
        "DeepWeeds", "Resisc45"
    ]
    data_indices_suffix = ""
    if any(x in dname for x in dataset_no_split):
        if mode is not None:
            data_indices_suffix = "_partial"
    return base, data_indices_suffix, mode


def get_image_folder(dataset_name, data_path=None):

    base_dataset_name, data_indices_suffix, mode = get_split(dataset_name)

    if base_dataset_name in additional_data.__dict__.keys():
        dset = additional_data.__dict__[base_dataset_name](data_path,
                                                           mode=mode)
    else:
        dset = ImageFolder(data_path)

    return dset, base_dataset_name, data_indices_suffix, mode


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 transform,
                 target_transform=identity,
                 dataset_name=None,
                 consecutive_label=False,
                 opt=None,
                 raise_error=True):
        self.transform = transform
        self.target_transform = target_transform
        self.raise_error = raise_error
        self.rng = np.random.RandomState(seed=opt.seed)

        self.data, base_dataset_name, data_indices_suffix, mode = get_image_folder(
            dataset_name, data_path)

        self.cls_to_idx = None
        if 'partial' in data_indices_suffix or 'disjoint' in data_indices_suffix or 'overlap' in data_indices_suffix:
            tmpfile = os.path.join(
                TMP_PATH, base_dataset_name +
                f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}.npy"
            )
            if not os.path.exists(tmpfile):
                prepare_data_indices(dataset_name, data_path, opt=opt)

            class_indices = np.load(tmpfile, allow_pickle=True).item()
            self.list_classes = list(class_indices.keys())
            self.indices = list(chain.from_iterable(class_indices.values()))

            if consecutive_label:
                self.cls_to_idx = {
                    c: i
                    for i, c in enumerate(sorted(self.list_classes))
                }
            loguru_log(f"loading indices from {tmpfile}")
        else:
            self.indices = None

    def __getitem__(self, i):
        idx = i
        if self.indices is not None:
            idx = self.indices[i]
        try:
            img, target = self.data[idx]
        except FileNotFoundError:
            if self.raise_error:
                raise FileNotFoundError

            rand_idx = int(self.rng.choice(len(self.data)))
            img, target = self.data[rand_idx]

        if self.cls_to_idx is not None:
            target = self.cls_to_idx[target]

        img = self.transform(img)
        target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.data)


def map_ind_to_label(dataset_name, data):
    tmpfile = os.path.join(TMP_PATH, dataset_name + f"_indices.npy")
    if not os.path.exists(tmpfile):
        sub_meta_indices = _get_ind_to_label(data, dataset_name)
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH, exist_ok=True)

        np.save(os.path.join(TMP_PATH, dataset_name + f"_indices.npy"),
                sub_meta_indices)


# def _get_ind_to_label(data, dataset_name=None):
#     sub_meta_indices = {}
#     for i, (_, label) in tqdm(enumerate(data),
#                               total=len(data),
#                               desc=f"storing indices {dataset_name}: "):
#         if label not in sub_meta_indices:
#             sub_meta_indices[label] = []
#         sub_meta_indices[label].append(i)
#     return sub_meta_indices


def _get_ind_to_label(data, dataset_name=None):
    sub_meta_indices = {}

    # Dummy dataset to be passed to DataLoader
    class LoaderInd:
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(data)

        def __getitem__(self, index):
            try:
                _, label = self.data[index]
            except FileNotFoundError:
                return None, None
            return label, index

    _loader = torch.utils.data.DataLoader(LoaderInd(data),
                                          batch_size=None,
                                          batch_sampler=None,
                                          collate_fn=identity,
                                          num_workers=60,
                                          shuffle=False)
    for label, i in tqdm(_loader,
                         total=len(data),
                         desc=f"storing indices {dataset_name}: "):
        if label is None:
            continue
        if label not in sub_meta_indices:
            sub_meta_indices[label] = []
        sub_meta_indices[label].append(i)

    return sub_meta_indices


def prepare_data_indices(dataset_name, data_path, opt=None):
    base_dataset_name, data_indices_suffix, mode = get_split(dataset_name)
    indfile = os.path.join(TMP_PATH, base_dataset_name + f"_indices.npy")

    if not os.path.exists(indfile):
        data, *_ = get_image_folder(dataset_name, data_path)
        map_ind_to_label(base_dataset_name, data)
    if data_indices_suffix:
        tmpfile = os.path.join(
            TMP_PATH, base_dataset_name +
            f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}.npy")
        if not os.path.exists(tmpfile):
            data_dict = np.load(indfile, allow_pickle=True).item()
            if "disjoint" in data_indices_suffix:
                helper.create_disjoint_indices(data_dict,
                                               base_dataset_name,
                                               num_split=4,
                                               min_way=opt.train_n_way,
                                               fraction=opt.split_fraction)

            elif "_sup" in data_indices_suffix or "_unsup" in data_indices_suffix or "partial" in data_indices_suffix:
                helper.create_partial_data(data_dict,
                                           base_dataset_name,
                                           fraction=opt.split_fraction)


@rank_zero_only
def loguru_log(msg):
    logger.info(msg)


class SetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 num_class,
                 batch_size,
                 transform,
                 dataset_name=None,
                 opt=None):

        base_dataset_name, data_indices_suffix, mode = get_split(dataset_name)
        if mode is None:
            mode = ""
            data_train = SimpleDataset(data_path,
                                       transform,
                                       dataset_name=base_dataset_name +
                                       "_train",
                                       opt=opt)
            data_test = SimpleDataset(data_path,
                                      transform,
                                      dataset_name=base_dataset_name + "_test",
                                      opt=opt)
            self.data = torch.utils.data.ConcatDataset((data_train, data_test))
        else:
            self.data = SimpleDataset(data_path,
                                      transform,
                                      dataset_name=base_dataset_name +
                                      f"_{mode}",
                                      opt=opt)

        tmpfile = os.path.join(
            TMP_PATH,
            base_dataset_name + f"{mode}_fs_indices_{data_indices_suffix}.npy")
        if not os.path.exists(tmpfile):
            self.sub_meta_indices = _get_ind_to_label(self.data,
                                                      base_dataset_name)
            np.save(tmpfile, self.sub_meta_indices)
        else:
            loguru_log(f"loading indices from {tmpfile}")
            self.sub_meta_indices = np.load(tmpfile, allow_pickle=True).item()

        self.cl_list = list(self.sub_meta_indices.keys())

        self.sub_dataloader = []

        self.gen = torch.Generator().manual_seed(opt.seed + 214743647)
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            generator=self.gen,
            num_workers=0,  #use main thread only or may receive multiple batches
            pin_memory=False)
        self.batch_size = batch_size

        # check if any data less than batch size
        rng = np.random.RandomState()
        for lab in self.sub_meta_indices:
            if len(self.sub_meta_indices[lab]) < batch_size:
                _orig = self.sub_meta_indices[lab]
                _needed = batch_size - len(_orig)
                _extra = rng.choice(_orig, size=_needed, replace=True)
                _new = np.concatenate((_orig, _extra), axis=0)
                self.sub_meta_indices[lab] = _new

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.data,
                                     self.sub_meta_indices[cl],
                                     cl,
                                     transform=None)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset,
                                            **sub_data_loader_params))

        loguru_log(
            f"loaded dataset {base_dataset_name}:: #class: {len(self.sub_meta_indices.keys())},"
            +
            f" #data: {sum(len(v) for _, v in self.sub_meta_indices.items())}")

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(self,
                 data_orig,
                 indices,
                 cl,
                 transform=transforms.ToTensor(),
                 target_transform=identity):
        self.sub_meta_indices = indices
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.data = data_orig

    def __getitem__(self, i):
        idx = self.sub_meta_indices[i]
        img, _ = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta_indices)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, seed=0):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            if self.n_classes >= self.n_way:
                yield torch.randperm(self.n_classes,
                                     generator=self.g)[:self.n_way]
            else:
                yield torch.from_numpy(
                    np.random.choice(self.n_classes, self.n_way))


def fn_divide(x):
    return x / 255.


class TransformLoader:
    def __init__(self,
                 image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                 normalize_type='imagenet',
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        if normalize_type == 'cifar10':
            self.normalize_param = dict(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        elif normalize_type == 'cifar100':
            self.normalize_param = dict(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761])
        else:  # imagenet
            self.normalize_param = dict(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    def parse_transform(self, transform_type):
        if transform_type == 'RandomColorJitter':
            return transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        elif transform_type == 'RandomGrayscale':
            return transforms.RandomGrayscale(p=0.2)
        elif transform_type == 'RandomGaussianBlur':
            return transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
        elif transform_type == 'RandomCrop':
            return transforms.RandomCrop(self.image_size, padding=4)
        elif transform_type == 'RandomResizedCrop':
            return transforms.RandomResizedCrop(self.image_size,
                                                scale=(0.2, 1.))
        elif transform_type == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)
        elif transform_type == 'Resize_up':
            return transforms.Resize(
                [int(self.image_size * 1.15),
                 int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return transforms.Normalize(**self.normalize_param)
        elif transform_type == 'Resize':
            return transforms.Resize(
                [int(self.image_size),
                 int(self.image_size)])
        else:
            method = getattr(transforms, transform_type)
            return method()

    def get_composed_transform(self, aug=False):

        if aug == 'MoCo':
            transform_list = [
                'RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale',
                'RandomGaussianBlur', 'RandomHorizontalFlip', 'ToTensor',
                'Normalize'
            ]
        elif aug is True or aug == 'true':
            transform_list = [
                'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                'Normalize'
            ]
        elif aug == 'few_shot_test':
            transform_list = [
                'Resize_up', 'CenterCrop', 'ToTensor', 'Normalize'
            ]
        elif aug == 'cifar_train':
            transform_list = [
                'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                'Normalize'
            ]
        elif aug == 'randaug' or aug == 'autoaug':
            transform_list = [
                'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                'Normalize'
            ]
        else:
            # transform_list = ['Resize', 'ToTensor', 'Normalize']
            transform_list = [
                'Resize_up', 'CenterCrop', 'ToTensor', 'Normalize'
            ]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)

        if aug == 'randaug':
            # from encoding.transforms.autoaug import RandAugment
            # transform.transforms.insert(0, RandAugment(2, 12))  # add RandAug in the beginning
            raise NotImplementedError("randAug not implemented yet")

        if aug in ['MoCo', 'randaug']:
            transform = TwoCropsTransform(transform)
        return transform


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self,
                 image_size,
                 batch_size,
                 dataset_name=None,
                 unlabel=False,
                 opt=None):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size,
                                            normalize_type=opt.normalize_type)
        self.dataset_name = dataset_name
        self.unlabel = unlabel

        self.opt = opt

    def get_data_loader(
            self,
            data_path,
            aug=True,
            return_data_idx=False,
            consecutive_label=False,
            limit_data=None,
            drop_last=True,
            shuffle=True):  #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        if isinstance(data_path, list):
            assert isinstance(self.dataset_name, list)
            assert len(data_path) == len(self.dataset_name)
            list_dataset = []
            for i, _ in enumerate(data_path):
                _dataset = self._get_dataset(data_path[i],
                                             transform,
                                             dataset_name=self.dataset_name[i])
                list_dataset.append(_dataset)
            # dataset = torch.utils.data.ConcatDataset(list_dataset)
            dataset = ConcatProportionDataset(list_dataset,
                                              return_data_idx=return_data_idx)
        else:
            dataset = self._get_dataset(data_path,
                                        transform,
                                        dataset_name=self.dataset_name,
                                        consecutive_label=consecutive_label)

        if limit_data and limit_data != 1:
            if limit_data <= 1:
                limit_len = int(len(dataset) * limit_data)
            else:
                limit_len = min(len(dataset), int(limit_data))
            rng = np.random.RandomState(seed=self.opt.seed)
            limit_indices = rng.choice(
                len(dataset),
                limit_len,
                replace=False if limit_len <= len(dataset) else True)
            dataset = torch.utils.data.Subset(dataset, limit_indices)

        loguru_log(
            f"loaded dataset {self.dataset_name}:: #data: {len(dataset)}")

        data_loader_params = dict(batch_size=self.batch_size,
                                  shuffle=shuffle,
                                  num_workers=self.opt.num_workers,
                                  pin_memory=False,
                                  drop_last=drop_last)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  **data_loader_params)

        return data_loader

    def _get_dataset(self,
                     data_path,
                     transform,
                     dataset_name,
                     consecutive_label=False):
        if self.unlabel is False:
            dataset = SimpleDataset(data_path,
                                    transform,
                                    dataset_name=dataset_name,
                                    consecutive_label=consecutive_label,
                                    opt=self.opt,
                                    raise_error=False)
        else:
            dataset = SimpleUnlabelDataset(data_path,
                                           transform,
                                           dataset_name=dataset_name)
        return dataset


class DistEpisodicBatchSampler(object):
    def __init__(self,
                 n_classes,
                 n_way,
                 n_episodes,
                 num_replicas=None,
                 rank=None):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

        # dist params
        self.num_replicas = num_replicas
        self.rank = rank

        self.g = torch.Generator()
        self.g.manual_seed(self.rank)

        # num sample to each replica
        self.num_samples = int(np.ceil(n_episodes / self.num_replicas))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield torch.randperm(self.n_classes, generator=self.g)[:self.n_way]


class SetDataManager(DataManager):
    def __init__(self,
                 data_path,
                 num_class,
                 image_size=224,
                 n_way=5,
                 n_support=5,
                 n_query=16,
                 n_episode=100,
                 aug=False,
                 dataset_name=None,
                 opt=None,
                 **kwargs):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.opt = opt
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

        transform = self.trans_loader.get_composed_transform(aug)
        self.dataset = SetDataset(data_path,
                                  num_class,
                                  self.batch_size,
                                  transform,
                                  dataset_name=dataset_name,
                                  opt=opt,
                                  **kwargs)

    def get_data_loader(
        self,
        aug=False,
        use_ddp=False,
        dist_args={},
    ):  #parameters that would change on train/val set

        if use_ddp is False:
            batch_sampler = EpisodicBatchSampler(len(self.dataset), self.n_way,
                                                 self.n_episode)
            worker_init_fn = lambda x: pl.seed_everything(self.opt.seed + int(
                x))
        else:
            batch_sampler = DistEpisodicBatchSampler(
                len(self.dataset),
                self.n_way,
                self.n_episode,
                num_replicas=dist_args["num_replicas"],
                rank=dist_args["rank"])
            worker_init_fn = lambda x: pl.seed_everything(dist_args["rank"] + x
                                                          )

        data_loader_params = dict(batch_sampler=batch_sampler,
                                  num_workers=self.opt.num_workers,
                                  pin_memory=False,
                                  worker_init_fn=worker_init_fn)
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  **data_loader_params)
        return data_loader
