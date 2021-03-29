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
    max_epochs="300"
fi

# NOTE chose appropriate mode
model="resnet50"

list_systems=("base_finetune" "moco" "moco_mit" "base_plus_moco" "supervised_mean2")

val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
    "Omniglot" "Resisc45" "SVHN" "DeepWeeds" "Kaokore" "Sketch" "Flowers102")

mode="train"

list_limit=("1")
SORT_MAX=false
METRIC="acc_mean"
list_seeds=("0")
launcher="local"

while getopts "s:m:l:x:b:e:d:c:v:" opt; do
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
    c)
        launcher="$OPTARG"
        ;;
    v)
        val_datasets=($OPTARG)
        ;;

    esac
done

if [ "$mode" = "show" ]; then
    echo "system,dataset,seed,score"
fi

for system in "${list_systems[@]}"; do
    model_name="${system}_${train_dataset}_${model}"

    if [ "$mode" = "" -o "$mode" = "train" ]; then
        if [ $system = "base_finetune" ]; then
            extra=""
        else
            extra=" "
        fi

        if [ "$train_dataset" = "ImageNet1K_train" ]; then
            extra="${extra} --val_dataset ImageNet1K_val "
        fi

        if [ "$launcher" = "slurm" ]; then
            slurm_cmd=" --is_slurm --slurm_log_root linear_results -n ${nodes} -g ${gpus} -t 6 -a --mem_per_cpu 10000"
        else
            slurm_cmd=""
        fi
        cmd="python main.py \
          --system ${system}  --dataset ${train_dataset} \
          --gpus -1 --model ${model} \
          --model_name ${model_name} \
          --num_nodes ${nodes} --num_workers 6 \
          --eval_mode linear --check_val_every_n_epoch 20 \
          --print_freq 10  --progress_bar_refresh_rate 10 \
          --moco_K ${moco_K} --max_epochs ${max_epochs} ${extra} ${slurm_cmd}"
        echo $cmd
        eval $cmd

    elif [ "$mode" = "tune" ]; then

        ############ Linear evaluation ##############
        for val_data in "${val_datasets[@]}"; do
            for limit_data in "${list_limit[@]}"; do
                for lr in "0.01" "0.001"; do
                    for bs in "128" "32"; do
                        for wd in "0" "1e-4" "1e-5"; do
                            extra=" "
                            suffix=""

                            if [ "$limit_data" != "1" ]; then
                                extra=" ${extra} --limit_train_data ${limit_data} "
                                suffix="${suffix}_lim_${limit_data}"
                            fi

                            slurm_cmd=" --is_slurm --slurm_log_root eval_linear_results -n 1 -g 1 -t 3 --mem_per_cpu 10000"

                            suffix="_HPlr_e_${lr}__batch_size_e_${bs}__optim_wd_e_${wd}HP"

                            # ckpt="none"
                            ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                            if [ ! -e "$ckpt" ]; then
                                echo "$ckpt does not exist"
                                continue
                            fi

                            max_epochs=20
                            # NOTE: model_name should be escaped properly with \" \"
                            _milestones=$(bc <<<"${max_epochs}/2"),$(bc <<<"${max_epochs}*3/4")
                            cmd="python main.py --system linear_eval \
                                --train_aug true --val_aug false \
                                --dataset ${val_data}_train --val_dataset ${val_data}_test \
                                --split_fraction 0.7 --gpus 1 \
                                --ckpt ${ckpt} --load_base --model ${model} \
                                --model_name \"linbn_${val_data}_${model_name}${suffix}\" \
                                --print_freq -1 --progress_bar_refresh_rate 0 \
                                --check_val_every_n_epoch 5 \
                                --batch_size ${bs} --lr ${lr} --optim_wd ${wd}  \
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
                    suff_tune="${suffix}"
                    if [ "$limit_data" != "1" ]; then
                        extra=" ${extra} --limit_train_data ${limit_data} "
                        suffix="${suffix}_lim_${limit_data}"
                    fi

                    slurm_cmd=" --is_slurm --slurm_log_root eval_linear_results -n 1 -g 1 -t 3  --mem_per_cpu 10000"

                    ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                    if [ ! -e "$ckpt" ]; then
                        echo "$ckpt does not exist"
                        continue
                    fi

                    list_suff=()
                    for lr in "0.01" "0.001"; do
                        for bs in "128" "32"; do
                            for wd in "0" "1e-4" "1e-5"; do
                                _suffix="${suff_tune}_HPlr_e_${lr}__batch_size_e_${bs}__optim_wd_e_${wd}HP"
                                list_suff+=("\"${_suffix}\"")
                            done
                        done
                    done

                    if [ $seed != "0" ]; then
                        extra="${extra} --seed ${seed}  --suffix _seed-${seed} "
                    fi

                    extra="${extra}  --hp_load true  --hp_suffixes ${list_suff[*]}"

                    max_epochs=50 # for hp tune
                    _milestones=$(bc <<<"${max_epochs}/2"),$(bc <<<"${max_epochs}*3/4")
                    cmd="python main.py --system linear_eval \
                        --train_aug true --val_aug false \
                        --dataset ${val_data}_train --val_dataset ${val_data}_test \
                        --split_fraction 0.7 --gpus 1 \
                        --ckpt ${ckpt} --load_base --model ${model} \
                        --model_name linbn_${val_data}_${model_name}${suffix} \
                        --print_freq -1 --progress_bar_refresh_rate 0 \
                        --check_val_every_n_epoch 5 \
                        --batch_size ${bs} --lr ${lr} \
                        --scheduler step  --step_lr_milestones ${_milestones} \
                        --linear_bn --linear_bn_affine false  \
                        --max_epochs ${max_epochs} ${extra} ${slurm_cmd}"
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
                    mname="linbn_${val_data}_${model_name}${suffix}"
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
                        # val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | grep -oP "\d*\.\d+" | sort -g | tail -1)
                        val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | grep -oP "\d*\.\d+" | tail -1)
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

# IN-1K evaluation
# for lr in "0.01"; do
#     nodes=5
#     slurm_cmd=" --is_slurm --slurm_log_root eval_linear_results -n ${nodes} -g ${gpus} -t 6 -a --mem_per_cpu 10000"
#     val_data="ImageNet1K_val"
#     ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"

#     max_epochs=100
#     _milestones=$(bc <<<"${max_epochs}/2"),$(bc <<<"${max_epochs}*3/4"),$(bc <<<"${max_epochs}*9/10")
#     cmd="python main.py --system linear_eval \
#         --train_aug true --val_aug false \
#         --dataset ImageNet1K_train --val_dataset ImageNet1K_val \
#         --gpus -1  --num_nodes ${nodes} \
#         --ckpt ${ckpt} --load_base \
#         --model_name IN_linear_bn_${val_data}_${model_name}_lr_${lr} \
#         --print_freq -1 --progress_bar_refresh_rate 0 \
#         --linear_bn --linear_bn_affine false \
#         --check_val_every_n_epoch 10 \
#         --scheduler step  --step_lr_milestones ${_milestones} \
#         --batch_size 32 --lr ${lr} --num_workers 6 \
#         --warm_up true --warm_epochs 5 \
#         --max_epochs ${max_epochs} ${slurm_cmd}"
#     echo "$cmd"
#     eval "$cmd"
# done
