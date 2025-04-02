# wandb login
# wandb offline
# wandb online
export WANDB_PROJECT=test
export HF_HOME="/tmp/hf"
MODEL_DIR="/tmp/hf/models/Qwen2.5-Coder-7B-Instruct"
DATASET_DIR="/tmp/hf/datasets/KodCode-V1-SFT-R1-10k"
OUTPUT_DIR="/tmp/hf/exp/sft-Qwen2.5-Coder-7B-Instruct"
EXP_RUN_NAME="exp-sft-7B-16k-8v100-zero3"


deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    sft.py \
    --model_name_or_path $MODEL_DIR \
    --dataset_name $DATASET_DIR \
    --do_train true \
    --do_eval false \
    --dataset_num_proc 16 \
    --max_length 16384 \
    --learning_rate 1.0e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1.0e-8 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --seed 42 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs "{\"use_reentrant\": false}" \
    --fp16 \
    --use_liger_kernel \
    --report_to wandb \
    --run_name $EXP_RUN_NAME \
    --deepspeed ./deepspeed/ds_zero3_offload.json
