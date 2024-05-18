GPUS=2
MODEL_PATH=/data/nlp/meta-llama/Llama-2-7b-chat-hf-trtllm/
mpirun -n ${GPUS} --allow-run-as-root python server_trtllm.py --model_path=${MODEL_PATH}