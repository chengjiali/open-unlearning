#!/bin/bash

export NCCL_P2P_DISABLE=1  # Disable P2P if needed

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "UNDIAL"
    "SatImp"
    "WGA"
)
cls=(
    "none"
    "per_token_superloss"
    "per_sample_superloss"
    "easy_to_hard"
    "hard_to_easy"
)
models=(
    zephyr-7b-beta
    Phi-3.5-mini-instruct
    phi-1_5
)
data_splits=(
    "cyber"
    "bio"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=2


for data_split in "${data_splits[@]}"; do

    for model in "${models[@]}"; do
        for trainer in "${trainers[@]}"; do

            if [[ "$model" == "zephyr-7b-beta" ]]; then
                TRAIN_CMD="CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/wmdp/default.yaml \
                trainer=${trainer} \
                model=${model} \
                data_split=${data_split} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no"
            else
                per_device_train_batch_size=4 # on two gpus would make effective batch size 32
                gradient_accumulation_steps=4

                TRAIN_CMD="CUDA_VISIBLE_DEVICES=2,4 accelerate launch --config_file configs/accelerate/default_config_2.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/wmdp/default.yaml \
                trainer=${trainer} \
                model=${model} \
                data_split=${data_split} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no"
            fi

            EVAL_CMD="CUDA_VISIBLE_DEVICES=4 python src/eval.py \
            experiment=eval/wmdp/default.yaml \
            model=${model} \
            data_split=${data_split}"

            for cl in "${cls[@]}"; do

                # CL = SuperLoss
                if [[ "$cl" == "per_token_superloss" || "$cl" == "per_sample_superloss" ]]; then
                    for lam in 0.1 1 10; do
                        task_name=${cl}/C_2_lam_${lam}/wmdp_${model}_${data_split}_${trainer}

                        if [ ! -f saves/unlearn/"${task_name}"/evals/LMEval_SUMMARY.json ]; then
                            if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                                echo "${task_name}" "Model Not Found"
                                
                                eval ${TRAIN_CMD} \
                                task_name=${task_name} \
                                trainer.cl.method=${cl} \
                                trainer.cl.lam=${lam} \
                                trainer.cl.C=2
                            fi

                            if [ -f saves/unlearn/"${task_name}"/model.safetensors ] || [ -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                                echo "${task_name}" "Eval Not Found"
                                
                                eval ${EVAL_CMD} \
                                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                                task_name=${task_name} \
                                paths.output_dir=saves/unlearn/${task_name}/evals
                            fi
                        fi
                    done

                # Easy to hard / hard to easy
                elif [[ "$cl" == "easy_to_hard" || "$cl" == "hard_to_easy" ]]; then
                    for metric in loss prob exact_mem extraction_strength; do
                        task_name=${cl}/${metric}/wmdp_${model}_${data_split}_${trainer}

                        if [ ! -f saves/unlearn/"${task_name}"/evals/LMEval_SUMMARY.json ]; then
                            if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                                echo "${task_name}" "Model Not Found"

                                eval ${TRAIN_CMD} \
                                task_name=${task_name} \
                                trainer.cl.method=${cl} \
                                trainer.cl.difficulty_metric=${metric} \
                                trainer.cl.data_name=wmdp \
                                trainer.cl.split=${data_split}
                            fi

                            if [ -f saves/unlearn/"${task_name}"/model.safetensors ] || [ -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                                echo "${task_name}" "Eval Not Found"
                                
                                eval ${EVAL_CMD} \
                                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                                task_name=${task_name} \
                                paths.output_dir=saves/unlearn/${task_name}/evals
                            fi
                        fi
                    done
                
                # No CL
                elif [[ "$cl" == "none" ]]; then 
                    task_name=${cl}/wmdp_${model}_${data_split}_${trainer}

                    if [ ! -f saves/unlearn/"${task_name}"/evals/LMEval_SUMMARY.json ]; then
                        if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                            echo "${task_name}" "Model Not Found"

                            eval ${TRAIN_CMD} \
                            task_name=${task_name} \
                            trainer.cl.method="none"
                        fi

                        if [ -f saves/unlearn/"${task_name}"/model.safetensors ] || [ -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                            echo "${task_name}" "Eval Not Found"
                            
                            eval ${EVAL_CMD} \
                            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                            task_name=${task_name} \
                            paths.output_dir=saves/unlearn/${task_name}/evals
                        fi
                    fi
                else
                    echo "Unsupported CL method"
                    exit 1
                fi
            done
        done
    done
done
