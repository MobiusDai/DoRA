#!/bin/sh

adapter=mora
model=LLaMA-7B                 # [LLaMA2-7B, LLaMA3-8B]
base_model=yahma/llama-7b-hf   # [meta-llama/Llama-2-7b-hf, meta-llama/Meta-Llama-3-8]

gpuid=0
micro_batch_size=4

rank=4
alpha=8

train_path=ft_train_dataset/commonsense_170k.json
model_path=trained_models/mlora-r$rank-a$alpha-3e4
results_path=results/mlora-r$rank-a$alpha-3e4


CUDA_VISIBLE_DEVICES=$gpuid python finetune.py \
    --base_model $base_model \
    --data_path $train_path \
    --output_dir $model_path \
    --batch_size 16  --micro_batch_size $micro_batch_size --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name $adapter \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $rank --lora_alpha $alpha --use_gradient_checkpointing


for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u commonsense_evaluate.py \
    --model $model \
    --adapter MoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path
done