# https://github.com/NVIDIA/TensorRT-LLM/tree/v0.8.0/examples/llama

# python convert_hf.py \
#     --model_dir <path to the HF folder> \
#     --output_dir <tmp file for trtllm checkpoint> \
#     --dtype <float16, bfloat16, float32> \
#     --tp_size <number of GPUs>

# trtllm-build \
#     --checkpoint_dir <tmp file for trtllm checkpoint> \
#     --output_dir <final path for the trtllm checkpoint> \
#     --gpt_attention_plugin <dtype from step above> \
#     --gemm_plugin <dtype from step above> \
#     --context_fmha <"enable" on A100+ GPUs and "disable" otherwise> \
#     --paged_kv_cache <"enable" on A100+ GPUs and "disable" otherwise> \
#     --max_input_len 4096 \
#     --max_output_len 512 \
#     --max_batch_size <desired batch size>
    
    
GPUS=2
HF_MODEL="meta-llama/Llama-2-7b-chat-hf"
HF_CHECKPOINT_PATH="/data/nlp/${HF_MODEL}"
TRTLLM_CHECKPOINT_PATH_TMP="/tmp/${HF_MODEL}-trtllm"
TRTLLM_CHECKPOINT_PATH="/data/nlp/${HF_MODEL}-trtllm"
mkdir -p ${TRTLLM_CHECKPOINT_PATH_TMP}
mkdir -p ${TRTLLM_CHECKPOINT_PATH}


python3 download_hf.py \
    --hf_path ${HF_MODEL} \
    --save_path ${HF_CHECKPOINT_PATH}
    
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 python3 convert_hf.py \
    --model_dir ${HF_CHECKPOINT_PATH} \
    --output_dir ${TRTLLM_CHECKPOINT_PATH_TMP} \
    --dtype bfloat16 \
    --tp_size ${GPUS}
    
trtllm-build \
    --checkpoint_dir  ${TRTLLM_CHECKPOINT_PATH_TMP} \
    --output_dir  ${TRTLLM_CHECKPOINT_PATH} \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --context_fmha disable \
    --paged_kv_cache disable \
    --max_input_len 4096 \
    --max_output_len 512 \
    --max_batch_size 5 \
    
cp ${HF_CHECKPOINT_PATH}/tokenizer.model ${TRTLLM_CHECKPOINT_PATH}/tokenizer.model
echo Your server folder ${TRTLLM_CHECKPOINT_PATH}

