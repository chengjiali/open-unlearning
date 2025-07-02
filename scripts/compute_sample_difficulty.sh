#!/bin/bash

export tokenizer_parallelism=true


## TOFU
model="Llama-3.1-8B-Instruct"
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    if [ ! -f saves/sample_difficulty/tofu/${forget_split}/extraction_strength.pt ]; then
        echo "TOFU ${forget_split} Difficulty Not Found"

        CUDA_VISIBLE_DEVICES=3 python src/compute_sample_difficulty.py \
        experiment=eval/tofu/default.yaml \
        task_name=compute_sample_difficulty/tofu/${forget_split} \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model}
    fi
done


## MUSE
model=Llama-2-7b-hf
data_splits=(
    "News"
    "Books"
)

for data_split in "${data_splits[@]}"; do
    if [ ! -f saves/sample_difficulty/muse/${data_split}/extraction_strength.pt ]; then
        echo "MUSE ${data_split} Difficulty Not Found"

        CUDA_VISIBLE_DEVICES=3 python src/compute_sample_difficulty.py \
        experiment=eval/muse/default.yaml \
        task_name=compute_sample_difficulty/muse/${data_split} \
        data_split=${data_split} \
        model=${model}
    fi
done


## WMDP
model=zephyr-7b-beta

data_splits=(
    "cyber"
    "bio"
)

for data_split in "${data_splits[@]}"; do

    if [ ! -f saves/sample_difficulty/wmdp/${data_split}/extraction_strength.pt ]; then
        echo "WMDP ${data_split} Difficulty Not Found"

        CUDA_VISIBLE_DEVICES=3 python src/compute_sample_difficulty.py \
        experiment=eval/wmdp/default.yaml \
        task_name=compute_sample_difficulty/wmdp/${data_split} \
        data_split=${data_split} \
        model=${model}
    fi
done
