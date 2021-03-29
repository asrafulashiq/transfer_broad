import numpy as np
import os
import argparse
# import backbone

# model_dict = dict(ResNet10=backbone.ResNet10)


def config_parser(parent=None):
    if parent is not None:
        parser = argparse.ArgumentParser(parents=[parent],
                                         conflict_handler='resolve')
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--version", type=int, default=None)

    parser.add_argument("--system_mode",
                        type=str,
                        default="2d",
                        choices=["2d", "3d"],
                        help="default 2d, 3d mode not supported yet")

    parser.add_argument('--model',
                        default='resnet50',
                        help='backbone architecture')

    parser.add_argument("--system",
                        type=str,
                        default="base_finetune",
                        help="the system/model to run")

    # ---------------------------------- dataset --------------------------------- #
    parser.add_argument('--dataset',
                        default='miniImageNet_train',
                        type=str,
                        help='training dataset')
    parser.add_argument('--val_dataset',
                        default='EuroSAT',
                        type=str,
                        help='val dataset to use')
    parser.add_argument('--train_aug',
                        type=str,
                        default='true',
                        choices=['true', 'false', 'MoCo', 'randaug'])
    parser.add_argument('--val_aug', type=str, default='few_shot_test')

    # -------------------------------- evaluation -------------------------------- #
    parser.add_argument("--eval_mode", type=str, default="linear")

    parser.add_argument('--freeze_backbone', default='true')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=["train", "save_features", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--load_base",
                        action="store_true",
                        help="load only base backbone")

    parser.add_argument("--suffix",
                        type=str,
                        default=None,
                        help="add suffix to model name")
    parser.add_argument("--gpus", type=str, default="-1")

    parser.add_argument("--resume",
                        action="store_true",
                        help="whether to resume training")
    parser.add_argument("--pretrained",
                        action="store_true",
                        help="use imagenet pretrained model")
    parser.add_argument("--test", action="store_true", help="test model")

    # ----------------------------------- train ---------------------------------- #
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--use_norm", action="store_true")

    parser.add_argument(
        '--num_classes',
        default=None,
        type=int,
        help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--max_epochs',
                        default=400,
                        type=int,
                        help='Stopping epoch')
    parser.add_argument("--moco_K", type=int, default=64000)
    parser.add_argument("--shuffle_val",
                        action="store_true",
                        help="shuffle validation data-loader")
    parser.add_argument("--checkpoint_show_epoch", type=int, default=None)

    # ------------------------------- for few-shot ------------------------------- #
    parser.add_argument('--train_n_way',
                        default=5,
                        type=int,
                        help='class num to classify for training')
    parser.add_argument('--test_n_way',
                        default=5,
                        type=int,
                        help='class num to classify for testing (validation) ')
    parser.add_argument(
        '--n_shot',
        default=5,
        type=int,
        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int)
    parser.add_argument("--num_episodes", type=int, default=600)

    # -------------------------------- log params -------------------------------- #
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--tb_log", action="store_true")
    parser.add_argument("--disable_validation", action="store_true")
    parser.add_argument("--distributed_backend", type=str, default=None)
    parser.add_argument("--print_val", action="store_true")
    parser.add_argument('--unsave_latest', default='false')
    parser.add_argument("--load_flexible", action="store_true")
    parser.add_argument('--print_freq',
                        type=int,
                        default=50,
                        help='print frequency')
    parser.add_argument("--disable_logfile", default='false')
    parser.add_argument("--drop_last", type=str, default="true")

    # -------------------------------  finetune ------------------------------ #
    parser.add_argument("--linear_bn",
                        action="store_true",
                        help="add a batchnorm layer before linear")
    parser.add_argument("--linear_bn_affine", type=str, default='false')

    parser.add_argument("--fine_tune_type", type=str, default="LR")
    parser.add_argument("--deep_finetune_lr_mod", type=float, default=1.0)
    parser.add_argument("--deep_finetune_wd", type=float, default=0)
    parser.add_argument("--deep_finetune_momentum", type=float, default=0.9)
    parser.add_argument("--deep_finetune_lr_class", type=float, default=1)
    parser.add_argument("--deep_finetune_epoch", type=int, default=100)
    parser.add_argument("--deep_finetune_batch_size", type=int, default=12)
    parser.add_argument("--deep_finetune_freeze_bn",
                        type=str,
                        default='false',
                        choices=['true', 'false'])
    parser.add_argument("--deep_finetune_lastk", type=int, default=1)
    parser.add_argument("--deep_finetune_use_norm", type=int, default=1)

    parser.add_argument("--ft_normalize",
                        default="true",
                        choices=["true", "false"],
                        help="normalize feature in few shot finetune")
    parser.add_argument("--build_linear_evaluator", type=str, default="true")

    # ------------------------- override lightning params ------------------------ #
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--num_sanity_val_steps", default=0, type=int)
    parser.add_argument("--scheduler", type=str, default='cosine')
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=50)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--row_log_interval", type=int, default=10)
    parser.add_argument("--sync_batchnorm", action="store_true")
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--tpu_cores", default=None)

    # -------------------------- script for data overlap ------------------------- #
    parser.add_argument("--split_fraction",
                        type=float,
                        default=0.7,
                        help="train split")
    parser.add_argument("--train_suffix",
                        type=str,
                        default=None,
                        choices=['partial', 'disjoint', 'overlap', ''])
    parser.add_argument("--val_suffix",
                        type=str,
                        default=None,
                        choices=['partial', 'disjoint', 'overlap', ''])
    parser.add_argument("--limit_train_data", type=float, default=None)

    # --------------------------------- optimizer -------------------------------- #
    parser.add_argument("--normalize_type",
                        default='imagenet',
                        choices=['imagenet', 'cifar10', 'cifar100'])
    parser.add_argument("--optimizer",
                        type=str,
                        default="SGD",
                        choices=['SGD', 'Adam', 'RMSProp', 'LARS'])

    parser.add_argument("--step_lr_milestones",
                        type=str,
                        default="300,350,400")
    parser.add_argument("--optim_momentum", type=float, default=0.9)
    parser.add_argument("--optim_wd", type=float, default=1e-4)
    parser.add_argument("--warm_up",
                        type=str,
                        default='true',
                        choices=['true', 'false'],
                        help="whether to lr warm-up")
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--eval_level",
                        type=str,
                        default="clip",
                        choices=["clip", "video"])

    # --------------------------------- hyper-parameter search -------------------------------- #
    parser.add_argument("--hp_load", type=str, default='false')
    parser.add_argument("--hp_suffixes", nargs="*", default=[], type=str)
    parser.add_argument("--hp_base_name", type=str, default=None)
    parser.add_argument("--hp_metric", type=str, default='acc_mean')
    parser.add_argument("--hp_mode", type=str, default='max')

    # ----------------------------- validation split for hp search ----------------------------- #
    parser.add_argument("--train_val_split",
                        type=float,
                        default=0.7,
                        help="train split for validation")
    parser.add_argument("--val_as_test",
                        type=str,
                        default='false',
                        choices=['true', 'false'],
                        help='whether to use validation as test')

    return parser