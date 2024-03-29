<div align="center">    

# A Broad Study on the Transferability of Visual Representations with Contrastive Learning
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2103.13517-green)](https://arxiv.org/abs/2103.13517)

</div>


This repository contains code for the paper: [A Broad Study on the Transferability of Visual Representations with Contrastive Learning](https://arxiv.org/abs/2103.13517)

## Prerequisites

- PyTorch 1.7
- pytorch-lightning 1.1.5

Install the required dependencies by: 

```
pip install -r environments/requirements.txt
```

## How to Run

### Download Datasets

The data should be located in `~/datasets/cdfsl` folder. To download all the datasets:

```
bash data_loader/download.sh 
```


### Training 

```
python main.py --system ${system}  --dataset ${train_dataset} --gpus -1 --model resnet50 
```
where `system` is one of `base_finetune`(ce), `moco` (SelfSupCon), `moco_mit` (SupCon), `base_plus_moco` (CE+SelfSupCon), or `supervised_mean2` (SupCon+SelfSupCon).

To know more about the cli arguments, see `configs.py`.

You can also run the training script by `bash scripts/run_linear_bn.sh -m train`.

### Evaluation

#### Linear evaluation

```
python main.py --system linear_eval \
  --train_aug true --val_aug false \
  --dataset ${val_data}_train --val_dataset ${val_data}_test \
  --ckpt ${ckpt} --load_base --batch_size ${bs} \
  --lr ${lr} --optim_wd ${wd}  --linear_bn --linear_bn_affine false \
  --scheduler step  --step_lr_milestones ${_milestones}
```

You can also run the evaluation script by `bash scripts/run_linear_bn.sh -m tune` to hyper-parameter tune, and then `bash scripts/run_linear_bn.sh -m test` to do linear-evaluation on the optimal hyper-parameter.

#### Few-shot
```
python main.py --system few_shot \
    --val_dataset ${val_data} \
    --load_base --test --model ${model} \
    --ckpt ${ckpt} --num_workers 4
```

You can also run the evaluation script by `bash scripts/run_fewshot.sh`.

#### Full-network finetuning
``` 
python main.py --system linear_transfer \
    --dataset ${val_data}_train --val_dataset ${val_data}_test \
    --ckpt ${ckpt} --load_base \
    --batch_size ${bs} --lr ${lr} --optim_wd ${wd} \
    --scheduler step  --step_lr_milestones ${_milestones} \
    --linear_bn --linear_bn_affine false \
    --max_epochs ${max_epochs}
```

You can also run the evaluation script by `bash scripts/run_transfer_bn.sh -m tune` to hyper-parameter tune, and then `bash scripts/run_transfer_bn.sh -m test` to do linear-evaluation on the optimal hyper-parameter.


## Pretrained models

- ImageNet pretrained models can be found [here](https://drive.google.com/drive/folders/1MXD47VqofZnfQU7iKHE0wL08HuTqGuaK?usp=sharing)

- mini-ImageNet pretrained models can be found [here](https://drive.google.com/drive/folders/13CVCdLRKtjo5h1Q-i0j8Be9IPA5GWb8P?usp=sharing)

You can also convert our pretrained checkpoint into torchvision resnet style checkpoint by `python utils/convert_to_torchvision_resnet.py -i [input ckpt] -o [output path]`

## Citation

If you find this repo useful for your research, please consider citing the paper:

```
@inproceedings{islam2021broad,
  title={A broad study on the transferability of visual representations with contrastive learning},
  author={Islam, Ashraful and Chen, Chun-Fu Richard and Panda, Rameswar and Karlinsky, Leonid and Radke, Richard and Feris, Rogerio},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8845--8855},
  year={2021}
}
```

## Acknowledgement

- [cdfsl-benchmark](https://github.com/IBM/cdfsl-benchmark)
