cd ../
ROOT_DIR=/home/wangzj/aliyun/MoE-LPR/LLaMA-Factory
MODEL_PATH=/home/nfs04/wangzj/models/Qwen1.5-1.8B
STAGE1_PATH=/home/nfs04/wangzj/checkpoints/moe/test/checkpoint-10
OUTPUT_DIR=/home/nfs04/wangzj/checkpoints/moe/test

export WANDB_DISABLED=true
deepspeed --num_gpus 3 --master_port=9902 src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $STAGE1_PATH \
    --finetuning_type moe \
    --lpr_loss_coef 0.1 \
    --train_only_router \
    --do_train \
    --dataset slimpajam_1b,ar_2b,de_2b,ru_2b \
    --max_samples 50000 \
    --generate_lang_mask \
    --preprocessing_num_workers 16 \
    --cutoff_len 512 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 500 \
    --save_only_model \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16