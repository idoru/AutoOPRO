set -eu

pip install datasets accelerate deepspeed wandb git+https://github.com/yikangshen/transformers.git bitsandbytes sentencepiece peft
git clone https://github.com/xfactlab/orpo.git
cd orpo
sed -i 's/num_processes: 2/num_processes: 4/' ./src/accelerate/fsdp.yaml
sed -i 's/num_processes: 2/num_processes: 4/' ./src/accelerate/ds2.yaml
sed -i 's/gradient_accumulation_steps: 1/gradient_accumulation_steps: 4/' ./src/accelerate/ds2.yaml
sed -i 's/--num_proc", default=8/--num_proc", default=4/' ./src/args.py
sed -i 's/gradient_checkpointing=True/gradient_checkpointing=False/' ./main.py


wandb login $WANDB_TOKEN
wandb init -p $WANDB_PROJECT
accelerate launch --config_file ./src/accelerate/ds2.yaml main.py \
    --lr $LEARNING_RATE \
    --warmup_steps 100 \
    --model_name $MODEL_ID \
    --data_name $DATASET \
    --num_train_epochs $EPOCH \
    --prompt_max_length 128 \
    --response_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_proc 1
cd $OUTPUT
cd */
huggingface-cli login --token $TOKEN
huggingface-cli upload $NEW_MODEL . .
