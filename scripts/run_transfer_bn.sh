train_dataset="ImageNet1K_train"

platform=$(uname -p)
if [ "${platform}" = "x86_64" ]; then
    nodes=3
    gpus=8
else
    nodes=4
    gpus=6
fi

if [ "$train_dataset" = "tieredImageNet" ]; then
    moco_K="44928"
    nodes=4
    max_epochs="500"
elif [ "$train_dataset" = "ImageNet1K_train" ]; then
    moco_K="64000"
    # nodes=4
    max_epochs="400"
    val_dataset="ImageNet1K_val"
else
    moco_K="16384"
    nodes=1
    max_epochs="700"
fi

# NOTE chose appropriate mode
model="resnet50"
# model="wide_resnet50_2"

list_systems=("base_finetune" "base_finetune_moco" "moco" "moco_mit" "base_plus_moco" "supervised_mean2")

# list_systems=("base_finetune_moco" "base_finetune_aug_two" "base_plus_moco_two")

val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
    "Omniglot" "Resisc45" "SVHN" "DeepWeeds" "Kaokore" "Sketch" "Flowers102")

mode="train"

list_limit=("1")
SORT_MAX=false
METRIC="acc_mean"
list_seeds=("0")
launcher="local"

while getopts "s:m:l:x:b:e:d:v:r:" opt; do
    case ${opt} in
    s)
        list_systems=($OPTARG)
        ;;
    m)
        mode="$OPTARG"
        ;;
    l)
        list_limit=($OPTARG)
        ;;
    x)
        SORT_MAX="$OPTARG"
        ;;
    b)
        model="$OPTARG"
        ;;
    e)
        METRIC="$OPTARG"
        ;;
    d)
        list_seeds=($OPTARG)
        ;;
    v)
        val_datasets=($OPTARG)
        ;;
    r)
        launcher="$OPTARG"
        ;;
    esac
done

if [ "$mode" = "show" ]; then
    echo "system,dataset,seed,score"
fi

for system in "${list_systems[@]}"; do
    model_name="${system}_${train_dataset}_${model}"

    if [ "$mode" = "tune" ]; then

        ############ Linear evaluation ##############
        for val_data in "${val_datasets[@]}"; do
            for limit_data in "${list_limit[@]}"; do
                for lr in "0.01" "0.001"; do
                    for bs in "128" "32"; do
                        for wd in "0" "1e-4" "1e-5"; do
                            extra=" "
                            suffix=""
                            extra=" "
                            suffix=""

                            extra=" --unfreeze_lastk -1 --linear_feature_multiplier 1 --unfreeze_warmup_epoch 0 \
                                        --linear_bn --linear_bn_affine true "

                            if [ "$limit_data" != "1" ]; then
                                extra=" ${extra} --limit_train_data ${limit_data} "
                                suffix="${suffix}_lim_${limit_data}"
                            fi

                            slurm_cmd=" --is_slurm --slurm_log_root eval_linear_results -n 1 -g 1 -t 3 --mem_per_cpu 10000"

                            suffix="_HPlr_e_${lr}__batch_size_e_${bs}__optim_wd_e_${wd}HP"

                            ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                            if [ ! -e "$ckpt" ]; then
                                echo "$ckpt does not exist"
                                continue
                            fi

                            max_epochs=20
                            # NOTE: model_name should be escaped properly with \" \"
                            _milestones=$(bc <<<"${max_epochs}/2"),$(bc <<<"${max_epochs}*3/4")
                            cmd="python main.py --system linear_transfer \
                                --train_aug true --val_aug false \
                                --dataset ${val_data}_train --val_dataset ${val_data}_test \
                                --split_fraction 0.7 --gpus 1 --num_workers 4 \
                                --ckpt ${ckpt} --load_base --model ${model} \
                                --model_name \"tfmbn_${val_data}_${model_name}${suffix}\" \
                                --print_freq -1 --progress_bar_refresh_rate 0 \
                                --check_val_every_n_epoch 5 \
                                --batch_size ${bs} --lr ${lr} --optim_wd ${wd} \
                                --scheduler step  --step_lr_milestones ${_milestones} \
                                --linear_bn --linear_bn_affine false \
                                --max_epochs ${max_epochs} ${extra} ${slurm_cmd} \
                                --val_as_test true --version 0"
                            echo "$cmd"
                            eval "$cmd"
                        done
                    done
                done
            done
        done

    elif [ "$mode" = "test" ]; then

        ############ Linear evaluation ##############

        for val_data in "${val_datasets[@]}"; do
            for limit_data in "${list_limit[@]}"; do
                for seed in "${list_seeds[@]}"; do

                    suffix=""
                    extra=""

                    extra=" --unfreeze_lastk -1 --linear_feature_multiplier 1 --unfreeze_warmup_epoch 0 \
                        --linear_bn --linear_bn_affine true "

                    more_suff=""

                    if [ "$limit_data" != "1" ]; then
                        extra=" ${extra} --limit_train_data ${limit_data} "
                        # suffix="${suffix}_lim_${limit_data}"
                        more_suff="${more_suff}_lim_${limit_data}"
                    fi

                    if [ "$launcher" = "slurm" ]; then
                        slurm_cmd=" --is_slurm --slurm_log_root eval_linear_results -n 1 -g 1 -t 3  --mem_per_cpu 10000"
                    else
                        slurm_cmd=""
                    fi

                    ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                    if [ ! -e "$ckpt" ]; then
                        echo "$ckpt does not exist"
                        continue
                    fi

                    list_suff=()
                    for lr in "0.01" "0.001"; do
                        for bs in "128" "32"; do
                            for wd in "0" "1e-4" "1e-5"; do
                                _suffix="${suffix}_HPlr_e_${lr}__batch_size_e_${bs}__optim_wd_e_${wd}HP"
                                list_suff+=("\"${_suffix}\"")
                            done
                        done
                    done

                    # if [ "system" = "base_finetune_aug_two" ]; then
                    #     extra="${extra} --hp_base_name linbn_${val_data}_base_finetune_${train_dataset}_${model}"
                    # fi

                    if [ $seed != "0" ]; then
                        extra="${extra} --seed ${seed}  --suffix _seed-${seed} "
                        more_suff="${more_suff}_seed-${seed}"
                    fi

                    extra="${extra}  --hp_load true  --hp_suffixes ${list_suff[*]}"
                    if [ ! -z "$more_suff" ]; then
                        extra="${extra} --suffix ${more_suff}"
                    fi

                    max_epochs=50 # for hp tune
                    _milestones=$(bc <<<"${max_epochs}/2"),$(bc <<<"${max_epochs}*3/4")
                    cmd="python main.py --system linear_transfer \
                        --train_aug true --val_aug false \
                        --dataset ${val_data}_train --val_dataset ${val_data}_test \
                        --split_fraction 0.7 --gpus 1 --num_workers 4 \
                        --ckpt ${ckpt} --load_base --model ${model} \
                        --model_name tfmbn_${val_data}_${model_name}${suffix} \
                        --print_freq -1 --progress_bar_refresh_rate 0 \
                        --check_val_every_n_epoch 5 \
                        --batch_size 128 --lr 0.01 \
                        --scheduler step  --step_lr_milestones ${_milestones} \
                        --linear_bn --linear_bn_affine false  \
                        --max_epochs ${max_epochs} ${extra}  ${slurm_cmd}"
                    echo "$cmd"
                    eval "$cmd"
                done
            done
        done
    elif [ "$mode" = "show" ]; then
        for val_data in "${val_datasets[@]}"; do
            for limit_data in "${list_limit[@]}"; do
                for seed in "${list_seeds[@]}"; do

                    suffix=""
                    more_suff=""

                    if [ "$limit_data" != "1" ]; then
                        more_suff="${more_suff}_lim_${limit_data}"
                    fi

                    # path
                    mname="tfmbn_${val_data}_${model_name}${suffix}"
                    logpath="logs/${mname}_${model}${more_suff}"

                    if [ "$seed" != "0" ]; then
                        logpath="${logpath}_seed-${seed}"
                    fi

                    if [ ! -e "$logpath" ]; then
                        printf "${system},${val_data},${lr},${bs},\n"
                        continue
                    fi

                    fullpath="${logpath}/$(ls -t ${logpath} | head -1)/log.txt"

                    # METRIC="validation_ece"
                    if [ "$METRIC"="validation_ece" -o "$SORT_MAX" = "true" ]; then
                        val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | grep -oP "\d*\.\d+" | sort -g | tail -1)
                        # val=$(grep -oP "\d*\.\d+" <<<"${val}")
                    else
                        val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | tail -1)
                        val=$(grep -oP "\d*\.\d+" <<<"${val}")
                    fi
                    printf "${system},${val_data},${seed},${val}\n"

                done
            done
        done

    fi
done
