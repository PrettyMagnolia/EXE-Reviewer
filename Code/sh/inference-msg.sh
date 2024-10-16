# use origin dataset inferring to get the focus information
mnt_dir="/home/codereview"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

#torch.distribution.launch
python -m torch.distributed.run --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_msg_explain.py  \
  --model_name_or_path microsoft/codereviewer \
  --output_dir ../../save/gen \
  --load_model_path ../../save/gen/checkpoint \
  --eval_file msg-test.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 1 \
  --mask_rate 0.15 \
  --save_steps 1800 \
  --beam_size 10 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input \
  --topk 1 \
  --generate_focus \
  --focus_len 10 \
