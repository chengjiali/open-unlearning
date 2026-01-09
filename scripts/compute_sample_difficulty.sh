#!/bin/bash

export tokenizer_parallelism=true

gpuid=0


trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "UNDIAL"
    "SatImp"
    "WGA"
    "CEU"
)

## TOFU
models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do

        task_name=tofu_${model}_${forget_split}

        if [ ! -f saves/sample_difficulty/tofu/${task_name}_embedding-entanglement.pt ]; then
            echo "TOFU ${task_name} Difficulty Not Found"

            CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
            experiment=eval/tofu/default.yaml \
            task_name=${task_name} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model}
        fi

        # Oracle difficulty using post-unlearning loss
        for trainer in "${trainers[@]}"; do
            task_name=tofu_${model}_${forget_split}_${trainer}_oracle

            if [ ! -f saves/sample_difficulty/tofu/${task_name}-loss.pt ]; then
                echo "TOFU ${task_name} Oracle Difficulty Not Found"

                CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
                experiment=eval/tofu/default.yaml \
                task_name=${task_name} \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/none/tofu_${model}_${forget_split}_${trainer}
            fi
        done
    done
done


# MUSE
model=Llama-2-7b-hf
data_splits=(
    "News"
    "Books"
)

for data_split in "${data_splits[@]}"; do
    task_name=muse_${model}_${data_split}

    if [ ! -f saves/sample_difficulty/muse/${task_name}_embedding-entanglement.pt ]; then
        
        echo "MUSE ${task_name} Difficulty Not Found"

        CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
        experiment=eval/muse/default.yaml \
        task_name=${task_name} \
        data_split=${data_split} \
        model=${model}
    fi

    # Oracle difficulty using post-unlearning loss
    for trainer in "${trainers[@]}"; do
        task_name=muse_${model}_${data_split}_${trainer}_oracle
        ckpt=/home/jiali/archive/robust_unlearning/saves/unlearn/muse_${model}_${data_split}_${trainer}_standard_42

        if [ ! -f saves/sample_difficulty/muse/${task_name}-loss.pt ]; then
            echo "MUSE ${task_name} Oracle Difficulty Not Found"

            CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
            experiment=eval/muse/default.yaml \
            task_name=${task_name} \
            data_split=${data_split} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${ckpt}
        fi
    done
done


## WMDP
models=(
    zephyr-7b-beta
    Phi-3.5-mini-instruct
    phi-1_5
)
data_splits=(
    "cyber"
    "bio"
)

for data_split in "${data_splits[@]}"; do

    for model in "${models[@]}"; do
        
        task_name=wmdp_${model}_${data_split}
        
        if [ ! -f saves/sample_difficulty/wmdp/${task_name}_embedding-entanglement.pt ]; then
            echo "WMDP ${task_name} Difficulty Not Found"

            CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
            experiment=eval/wmdp/default.yaml \
            task_name=${task_name} \
            data_split=${data_split} \
            model=${model}
        fi

        # Oracle difficulty using post-unlearning loss
        for trainer in "${trainers[@]}"; do
            task_name=wmdp_${model}_${data_split}_${trainer}_oracle

            ckpt=/home/jiali/archive/unlearning_checkpoint/none/wmdp_${model}_${data_split}_${trainer}

            if [ ! -f saves/sample_difficulty/wmdp/${task_name}-loss.pt ]; then
                echo "WMDP ${task_name} Oracle Difficulty Not Found"

                CUDA_VISIBLE_DEVICES=$gpuid python src/compute_sample_difficulty.py \
                experiment=eval/wmdp/default.yaml \
                task_name=${task_name} \
                data_split=${data_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${ckpt}
            fi
        done
    done
done
