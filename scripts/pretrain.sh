cd ../runner/docmamba
conda activate docmamba

export NGPU=8
export NNODES=1

export model_name_or_path=../libs/configs/docmamba
export output_dir=../experiments/pretrain/
export dataloader_num_workers=2
export gradient_accumulation_steps=2
export max_steps=134000
export save_steps=2000
export learning_rate=5e-5
export warmup_ratio=0.1
export master_port=10025
export per_device_train_batch_size=1

if [[ $NNODES -gt 1 ]]; then
    python -m torch.distributed.launch --use-env --nproc_per_node $NGPU --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        pretrain.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --logging_strategy steps \
        --logging_steps 50 \
        --per_device_train_batch_size $per_device_train_batch_size \
        --learning_rate $learning_rate \
        --max_steps $max_steps \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio $warmup_ratio \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --ignore_data_skip True \
        --fp16
else
    python -m torch.distributed.launch --use-env --nproc_per_node=$NGPU --master_port=$master_port \
        pretrain.py \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --logging_strategy steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --logging_steps 50 \
        --learning_rate $learning_rate \
        --max_steps $max_steps \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio $warmup_ratio \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --ignore_data_skip True \
        --fp16
fi