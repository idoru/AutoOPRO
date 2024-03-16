pip install datasets accelerate wandb transformers bitsandbytes
git clone https://github.com/xfactlab/orpo.git
curl -O -L https://gist.githubusercontent.com/abideenml/fc9edf20f51b27310c2f0faa63007fc8/raw/wandb.py
python wandb.py --WANDB $WANDB_TOKEN
cd orpo
accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr $LEARNING_RATE \
    --warmup_steps 100 \
    --model_name $MODEL_ID \
    --data_name $DATASET \
    --num_train_epochs $EPOCHS \
    --prompt_max_length 128 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_proc 1
