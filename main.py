import os
from utils.custom_logger import CustomLogger
from utils.custom_callbacks import CheckpointCallback
import torch
import pytorch_lightning as pl
import argparse
from data_loader.data_module import DataModule
import platform
from helper import get_parser
from helper.helper_slurm import run_cluster, slurm_parser
from helper.helper_search import get_best_hp, update_params


def main(params, LightningSystem, *args, **kwargs):
    from helper import config_init, refine_args
    params = config_init(params)
    params = refine_args(params)

    if params.hp_load == 'true':
        if params.hp_base_name is None:
            glob_path = os.path.join(params.log_path, params.base_model_name)
        else:
            glob_path = os.path.join(params.log_path, params.hp_base_name)
        best_hp, _ = get_best_hp(glob_path,
                                 params.hp_suffixes,
                                 model=params.model,
                                 field=params.hp_metric,
                                 mode=params.hp_mode)
        params = update_params(params, best_hp)

    logger = CustomLogger(config=params)

    datamodule = DataModule(params)
    model = LightningSystem(params, datamodule)

    if params.ckpt is not None and params.ckpt != 'none':
        if params.load_base and (params.system != 'base_finetune'
                                 or 'linear_eval' in params.system
                                 or 'linear_transfer' in params.system):
            model.load_base(params.ckpt)
        else:
            ckpt = torch.load(
                params.ckpt,
                map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(ckpt, strict=not params.load_flexible)

    # callbacks
    checkpoint_callback = CheckpointCallback(
        dirpath=params.save_path,
        save_last=True,
        save_top_k=1,
        monitor='acc_mean',
        mode='max',
        save_weights_only=False,  # for hpc mode
        verbose=True,
        period=1)

    trainer = pl.Trainer.from_argparse_args(
        params,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[],
        weights_save_path=params.save_path,
        replace_sampler_ddp=False,  # for hpc
        limit_test_batches=params.limit_val_batches,
        benchmark=False  # cudnn benchmark
    )

    if params.mode == 'test':
        trainer.test(model)
    if params.mode == 'train':
        trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_slurm",
                        action="store_true",
                        help="whether to run in slurm")
    args = parser.parse_known_args()[0]

    if args.is_slurm is True:
        # submit job to slurm
        parser, lt_system = get_parser(parent=parser)
        parser = slurm_parser(parser)
        run_cluster(parser, main, lt_system)
    else:
        # add Lightning parse
        parser, lt_system = get_parser(parent=parser)
        params = parser.parse_args()
        main(params, lt_system)