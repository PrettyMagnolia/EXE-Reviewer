# batch size 6 for 16 GB GPU

mnt_dir="/home/codereview"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO



# Change the arguments as required:
#   model_name_or_path: the path of the model to be finetuned (CodeReviewer Model)
#   train_filename: the path of the training file (after preprocess)
#   dev_filename: the path of the development file (after preprocess)
#   output_dir: the directory to save finetuned model
#   focus_len: the length of the focuses (equal to the interface script)
#   causal_seq_model_path: the path of the causal model for sequence
#   causal_tok_model_path: the path of the causal model for token

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_msg_explain.py  \
    --train_epochs 30 \
    --model_name_or_path microsoft/codereviewer \
    --output_dir ../../save/gen \
    --train_filename msg-train-preprocess.jsonl \
    --dev_filename msg-valid-preprocess.jsonl \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 6 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 60000 \
    --gpu_per_node $PER_NODE_GPU \
    --node_index $RANK \
    --seed 2233 \
    --raw_input True \
    --save_interval_epochs 100 \
    --has_focus \
    --focus_len 10 \
    --has_explain \
    --causal_seq_model_path ../model/unicausal-seq-baseline \
    --causal_tok_model_path ../model/unicausal-tok-baseline \
