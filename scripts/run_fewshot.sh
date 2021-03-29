model="resnet18"
# model="wide_resnet50_2"

# list_systems=("base_finetune" "moco" "moco_mit" "base_plus_moco"
#     "supervised_mean2" "base_finetune_aug_two")

list_systems=("base_finetune" "moco" "moco_mit" "base_plus_moco"
    "supervised_mean2")

# list_systems=("base_finetune" "base_finetune_moco" "base_plus_moco_two" "base_finetune_aug_two")

val_datasets=("miniImageNet_test" "EuroSAT" "CropDisease" "ChestX" "ISIC" "DTD"
    "Omniglot" "Resisc45" "SVHN" "DeepWeeds" "Kaokore" "Sketch" "Flowers102")

mode="test"
list_fmethod=("LR")
norm="true"

n_shot="5"
launcher="local" # slurm

while getopts "s:m:f:n:z:l:v:" opt; do
    case ${opt} in
    s)
        list_systems=($OPTARG)
        ;;
    m)
        mode="$OPTARG"
        ;;
    f)
        list_fmethod=($OPTARG)
        ;;
    n)
        norm=$OPTARG
        ;;
    z)
        n_shot=$OPTARG
        ;;
    l)
        launcher="$OPTARG"
        ;;
    v)
        val_datasets=($OPTARG)
        ;;
    esac
done

if [ "$mode" = "show" ]; then
    echo "system,dataset,std,score"
fi

for system in "${list_systems[@]}"; do
    # model_name="${system}_ImageNet1K_train_${model}"
    if [ "$mode" = "test" -o "$mode" = "" ]; then
        ############ Few shot evaluation ##############
        for fmethod in "${list_fmethod[@]}"; do
            extra=" "
            suffix="_${fmethod}"

            if [ "$fmethod" = "LR" ]; then
                time="00:20:00"
            else
                time="00:30:00"
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

            if [ "$launcher" = "slurm" ]; then
                slurm_cmd=" --is_slurm --slurm_log_root eval_fewshot_results -n 1 -g 1 -t "$time" --cpus_per_task 4 --mem_per_cpu 10000"
            else
                slurm_cmd=""
            fi
            for val_data in "${val_datasets[@]}"; do
                model_name="${system}_miniImageNet_train_${model}"
                ckpt="ckpt/cci/${model_name}_${model}/last.ckpt"

                if [ ! -e $ckpt ]; then
                    echo "$ckpt does not exist"
                    continue
                fi

                cmd="python main.py --system few_shot \
                        --val_dataset ${val_data} \
                        --load_base --test --model ${model} \
                        --ckpt ${ckpt} --num_workers 4 \
                        --model_name few_${val_data}_${model_name}${suffix} \
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

            for val_data in "${val_datasets[@]}"; do
                # path
                model_name="${system}_miniImageNet_train_${model}"
                mname="few_${val_data}_${model_name}${suffix}"
                logpath="test_logs/${mname}_${model}_${val_data}"
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

                # get std
                std=$(grep -oP "\+\-\W+\d*\.\d+" ${fullpath} | tail -1)
                std=$(grep -oP "\d*\.\d+" <<<"${std}")
                val=$(bc <<<"${val}*100" | xargs printf "%0.2f")
                std=$(bc <<<"${std}*100" | xargs printf "%0.2f")
                printf "${system},${val_data},${std},${val}\n"

            done
        done
    fi
done
