train_dataset="miniImageNet_train"
model="resnet18"

# test on lm_moco in base_plus_moco
# system="base_plus_moco"
val_datasets="CropDisease,EuroSAT,ISIC,DTD,DeepWeeds,SVHN"

# list_systems=("base_finetune" "moco" "moco_mit" "base_plus_moco" "supervised_mean2")
list_systems=("base_finetune_moco" "base_finetune_aug_two" "base_plus_moco_two")

mode="train"
list_fmethod=("LR")
n_shot="5"
norm="true"
self="false"
with_std="false"
launcher="slurm"

while getopts "s:t:m:z:v:b:x:d:c:" opt; do
    case ${opt} in
    s)
        list_systems=($OPTARG)
        ;;
    t)
        train_dataset="$OPTARG"
        ;;
    m)
        mode=$OPTARG
        ;;
    z)
        n_shot=$OPTARG
        ;;
    v)
        val_datasets="$OPTARG"
        ;;
    b)
        model="$OPTARG"
        ;;
    x)
        self="$OPTARG"
        ;;
    d)
        with_std="$OPTARG"
        ;;
    c)
        launcher="$OPTARG"
        ;;
    esac
done

nodes=1

if [[ "$mode" = "show"* ]]; then
    echo "system,dataset,lr,bs,score"
fi

for system in "${list_systems[@]}"; do
    extra=""
    suffix=""

    model_name="${system}_${train_dataset}_${model}"

    if [ "$mode" = "train" ]; then
        slurm_cmd=" --is_slurm --slurm_log_root eval_fewshot_results -n ${nodes} -g 4 -t 6 -a --cpus_per_task 4 --mem_per_cpu 15000"

        extra="${extra} --ft_normalize true "

        if [ $system = "supervised_mean2" ]; then
            extra="${extra} --lm_intra 1 "
        elif [[ $system = "base_plus_moco"* ]]; then
            extra="${extra} --lm_moco 2 "
        else
            extra=""
            suffix=""
        fi

        if [[ "$train_dataset" = "ImageNet100"* ]]; then
            extra="${extra} --moco_K 64000 "
            nodes=2
            slurm_cmd=" --is_slurm --slurm_log_root eval_fewshot_results -n ${nodes} -g 6 -t 6 -a \
                    --cpus_per_task 4 --mem_per_cpu 15000"
        elif [[ "$train_dataset" = "tieredImageNet"* ]]; then
            extra="${extra} --moco_K 64000 "
            nodes=2
            slurm_cmd=" --is_slurm --slurm_log_root eval_fewshot_results -n ${nodes} -g 6 -t 6 -a \
                    --cpus_per_task 4 --mem_per_cpu 15000"
        fi

        if [ "$launcher" != "slurm" ]; then
            slurm_cmd=""
        fi

        cmd="python main.py --system ${system} --dataset ${train_dataset} \
            --val_dataset ${val_datasets} --num_episodes 400 --eval_mode few_shot \
            --model_name ${model_name}${suffix} --num_nodes ${nodes} \
            --moco_K 16384 --check_val_every_n_epoch 20 --model ${model} \
            --print_freq -1 --progress_bar_refresh_rate 0 \
            --batch_size 32 --lr 0.01 --gpus -1 --max_epochs 300 \
            ${extra} ${slurm_cmd}"

        echo $cmd
        eval "$cmd"

    elif [ "$mode" = "test_linear" ]; then
        for limit_data in "1"; do             #"0.5" "0.25" "3000"; do
            for lr in "0.1" "1" "10" "30"; do #"1" "10" "30"; do
                for bs in "32" "256"; do
                    if [[ $system = "base_"* ]]; then
                        # lr=$(bc <<<"scale=10;${lr}*0.1") # do not need large lr for base finetune
                        if [ "$lr" = "30" ]; then lr="0.01"; fi
                    fi

                    extra=" "
                    suffix="_lr_${lr}_bs_${bs}"

                    slurm_cmd=" --is_slurm c--slurm_log_root eval_linear_results -n 1 -g 1 -t 2 -a --mem_per_cpu 10000"

                    val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
                        "Omniglot" "Resisc45" "SVHN" "DeepFashion" "DeepWeeds" "ExDark" "Kaokore" "Sketch"
                        "CIFAR100" "CUB" "Flowers102" "Pets")

                    for val_data in "${val_datasets[@]}"; do
                        ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                        if [ ! -e "$ckpt" ]; then
                            echo "$ckpt does not exist"
                            exit
                        fi
                        if [ $system = "base_plus_moco_intra" ]; then
                            ckpt="ckpt/cci/base_plus_moco_ImageNet1K_train_${model}_intra_${model}/last.ckpt"
                        fi
                        cmd="python main.py --system linear_eval \
                        --train_aug true --val_aug false \
                        --dataset ${val_data}_train --val_dataset ${val_data}_test \
                        --split_fraction 0.7 --gpus 1 \
                        --ckpt ${ckpt} --load_base \
                        --model_name eval_${val_data}_${model_name}${suffix} \
                        --print_freq -1 --progress_bar_refresh_rate 0 \
                        --check_val_every_n_epoch 10 \
                        --batch_size ${bs} --lr ${lr} \
                        --max_epochs 100 ${extra} ${slurm_cmd}"
                        echo "$cmd"
                        eval "$cmd"
                    done
                done
            done
        done

    elif [ "$mode" = "show_linear" ]; then

        for limit_data in "1"; do             #"0.5" "0.25" "3000"; do
            for lr in "0.1" "1" "10" "30"; do #"1" "10" "30"; do
                for bs in "32" "256"; do
                    if [[ $system = "base_"* ]]; then
                        # lr=$(bc <<<"scale=10;${lr}*0.1") # do not need large lr for base finetune
                        if [ "$lr" = "30" ]; then lr="0.01"; fi
                    fi

                    extra=" "
                    suffix="_lr_${lr}_bs_${bs}"

                    val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
                        "Omniglot" "Resisc45" "SVHN" "DeepFashion" "DeepWeeds" "ExDark" "Kaokore" "Sketch"
                        "CIFAR100" "CUB" "Flowers102" "Pets")
                    for val_data in "${val_datasets[@]}"; do
                        # path
                        mname="eval_${val_data}_${model_name}${suffix}"
                        logpath="logs/${mname}_${model}"
                        if [ ! -e $logpath ]; then
                            printf "${system},${val_data},${lr},${bs},\n"
                            continue
                        fi
                        fullpath="${logpath}/$(ls -t ${logpath} | head -1)/log.txt"
                        blue=$(tput setaf 4)
                        normal=$(tput sgr0)
                        METRIC="acc_mean"

                        # METRIC="validation_ece"
                        val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | tail -1)
                        val=$(grep -oP "\d*\.\d+" <<<"${val}")
                        printf "${system},${val_data},${lr},${bs},${val}\n"
                    done
                done
            done
        done

    elif [ "$mode" = "test" ]; then
        ############ Few shot evaluation ##############
        for fmethod in "${list_fmethod[@]}"; do
            extra="${extra} --ft_normalize true "
            suffix="_${fmethod}"

            if [ "$fmethod" = "LR" ]; then
                time="00:20:00"
            else
                time="00:40:00"
            fi

            if [ "${n_shot}" -gt "5" ]; then
                time="00:45:00"
            fi

            if [ "$norm" = "false" ]; then
                extra="${extra} --ft_normalize false "
                suffix="${suffix}_nnorm"
            fi

            if [ ${n_shot} != "5" ]; then
                extra="${extra} --n_shot ${n_shot} "
                suffix="${suffix}_shot-${n_shot}"
            fi

            slurm_cmd=" --is_slurm --slurm_log_root eval_fewshot_results -n 1 -g 1 -t "$time" --cpus_per_task 4 --mem_per_cpu 15000"
            # val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
            #     "Omniglot" "Resisc45" "SVHN" "DeepFashion" "DeepWeeds" "ExDark" "Kaokore" "Sketch"
            #     "CIFAR100" "CUB" "Flowers102" "Pets")
            val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
                "Omniglot" "Resisc45" "SVHN" "DeepWeeds" "Kaokore" "Sketch" "Flowers102")

            if [[ "$train_dataset" = "miniImageNet"* ]]; then val_datasets+=("miniImageNet_test"); fi
            if [[ "$train_dataset" = "tieredImageNet"* ]]; then val_datasets+=("tieredImageNet_test"); fi
            # if [[ "$train_dataset" = "ImageNet100"* ]]; then val_datasets+=("ImageNet100_val"); fi

            for val_data in "${val_datasets[@]}"; do
                # model_name="${system}_ImageNet1K_train_${model}"
                ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"
                if [ ! -e $ckpt ]; then
                    echo "$ckpt does not exist"
                    continue
                fi

                cmd="python main.py --system few_shot \
                        --val_dataset ${val_data} \
                        --load_base --test \
                        --ckpt ${ckpt} --num_workers 4 --model ${model} \
                        --model_name fs_${val_data}_${model_name}${suffix} \
                        --print_freq -1 --progress_bar_refresh_rate 0 \
                        --print_val ${extra} ${slurm_cmd}"
                echo "$cmd"
                eval "$cmd"
            done
        done
    elif [ "$mode" = "show" ]; then
        for fmethod in "${list_fmethod[@]}"; do
            suffix="_${fmethod}"
            if [ "$norm" = "false" ]; then
                extra="${extra} --ft_normalize false "
                suffix="${suffix}_nnorm"
            fi

            if [ ${n_shot} != "5" ]; then
                extra="${extra} --n_shot ${n_shot} "
                suffix="${suffix}_shot-${n_shot}"
            fi

            # val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
            #     "Omniglot" "Resisc45" "SVHN" "DeepFashion" "DeepWeeds" "ExDark" "Kaokore" "Sketch"
            #     "CIFAR100" "CUB" "Flowers102" "Pets")
            val_datasets=("EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
                "Omniglot" "Resisc45" "SVHN" "DeepWeeds" "Kaokore" "Sketch" "Flowers102")

            if [ "$self" = "true" ]; then
                if [[ "$train_dataset" = "miniImageNet"* ]]; then val_datasets=("miniImageNet_test"); fi
                if [[ "$train_dataset" = "tieredImageNet"* ]]; then val_datasets=("tieredImageNet_test"); fi
            fi

            for val_data in "${val_datasets[@]}"; do
                # path
                # model_name="${system}_ImageNet1K_train_${model}"
                mname="fs_${val_data}_${model_name}${suffix}"
                logpath="test_logs/${mname}_${model}_${val_data}"
                if [ ! -e $logpath ]; then
                    printf "${system},${val_data},-,-,\n"
                    continue
                fi
                fullpath="${logpath}/$(ls -t ${logpath} | head -1)/log.txt"
                blue=$(tput setaf 4)
                normal=$(tput sgr0)
                METRIC="acc_mean"
                # METRIC="validation_ece"
                val=$(grep -oP "${METRIC}\W+\d*\.\d+" ${fullpath} | tail -1)
                val=$(grep -oP "\d*\.\d+" <<<"${val}")

                # get std
                if [ "${with_std}" = "true" ]; then
                    std=$(grep -oP "\+\-\W+\d*\.\d+" ${fullpath} | tail -1)
                    std=$(grep -oP "\d*\.\d+" <<<"${std}")
                    val=$(bc <<<"${val}*100" | xargs printf "%0.2f")
                    std=$(bc <<<"${std}*100" | xargs printf "%0.2f")
                    printf "${system},${val_data},-,-,${val}+-${std} \n"
                    continue
                fi

                printf "${system},${val_data},-,-,${val}\n"
            done
        done
    fi

done
