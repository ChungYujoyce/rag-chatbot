HF_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
HF_CHECKPOINT_PATH="/home/joyce/model/${HF_MODEL}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server  \
    --model=${HF_CHECKPOINT_PATH} \
    --served-model-name='llama3-70b-instruct' \
    --max-model-len=8192 \
    --block-size=8 \
    --gpu-memory-utilization=1 \
    --max-num-seqs=1 \
    --tensor-parallel-size=1 \
    --pipeline-parallel-size=1 \
    --trust-remote-code \
    --dtype bfloat16 \