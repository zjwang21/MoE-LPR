cd ../
ROOT_DIR=/home/wangzj/aliyun/temp_data/LLaMA-Factory
MODEL_PATH=/home/nfs04/wangzj/models/Qwen1.5-1.8B
CACHE_PATH=/home/nfs03/wangzj/dataset/pretrain/arderu6b
OUTPUT_DIR=/home/nfs04/wangzj/checkpoints/moe/test

export WANDB_DISABLED=true
deepspeed --num_gpus 4 --master_port=9902 src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --finetuning_type moe \
    --topk 2 \
    --moe_num_experts 4 \
    --aux_loss_coef 0.01 \
    --do_train \
    --dataset_dir $ROOT_DIR/data \
    --cache_path $CACHE_PATH \
    --preprocessing_num_workers 16 \
    --cutoff_len 512 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 10 \
    --save_only_model \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16