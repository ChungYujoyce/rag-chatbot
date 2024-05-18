# https://github.com/Kipok/NeMo-Skills/blob/main/nemo_skills/inference/server/serve_trt.py

import json
import logging
logging.basicConfig(level=logging.INFO)
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import tensorrt_llm
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from tensorrt_llm.runtime import ModelRunnerCpp
from mpi4py import MPI
from transformers import AutoTokenizer


class TritonServerGenerate(Resource):
    def __init__(self, model):
        self.model = model
        self.comm = MPI.COMM_WORLD

    def generate(
        self,
        prompts,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        output = self.model.forward(
            prompts,
            max_output_token=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )
        return output

    def post(self):
        logging.info("request IP: " + str(request.remote_addr))
        logging.info(json.dumps(request.get_json()))

        input_request = request.get_json()
        
        model = input_request["model"]
        prompts = input_request["prompt"]
        tokens_to_generate = input_request.get("max_tokens", 64)
        random_seed = input_request.get("seed", 0)
        stop_words_list = input_request.get("stop", [])
        if '<|eot_id|>' not in stop_words_list:
            stop_words_list.append('<|eot_id|>')
        
        temperature = input_request.get("temperature", 1.0)
        top_p = input_request.get("top_p", 0.95)
        top_k = input_request.get("top_k", 40)
        
        frequency_penalty = 0.0
        repetition_penalty = 1.2
        
        data = dict(
            prompts=prompts,
            max_new_tokens=tokens_to_generate,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )
        self.comm.Barrier()
        data = self.comm.bcast(data, root=0)

        out = self.generate(**data)
        response = {
            "choices": [
                {"text": o['text'].strip('<|eot_id|>')} for o in out
            ],
            "usage": {
                "completion_tokens": out[0]['output_length'],
                "prompt_tokens": out[0]['input_length'],
                "total_tokens": out[0]['input_length'] + out[0]['output_length'],
            }
        }
        logging.info(json.dumps(response["usage"]))
        return jsonify(response)


def parse_input(input_texts: str, tokenizer):
    batch_input_ids = [
        tokenizer.encode(
            input_text,
            add_special_tokens=True,  # TODO: does this need to be true?
        )
        for input_text in input_texts
    ]
    batch_input_ids = [torch.tensor(x, dtype=torch.int32, device="cuda") for x in batch_input_ids]
    input_lengths = [x.size(0) for x in batch_input_ids]

    return batch_input_ids, input_lengths


def get_output(output_ids, input_lengths, max_output_len, tokenizer, eos_token):
    num_beams = output_ids.size(1)
    assert num_beams == 1
    output_texts = []
    for idx, input_len in enumerate(input_lengths):
        output_begin = input_len
        output_end = input_len + max_output_len
        outputs = output_ids[idx][0][output_begin:output_end]
        eos_ids = (outputs == eos_token).nonzero(as_tuple=True)[-1]
        if len(eos_ids) > 0:
            outputs = outputs[: eos_ids[0]]
        outputs = outputs.tolist()
        output_texts.append({
            'text': tokenizer.decode(outputs),
            'input_length': input_len,
            'output_length': len(outputs),
        })
    return output_texts


def prepare_stop_words(stop_words_list, tokenizer):
    # adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/b310ec675145c9ee7668592549f733df4abf1e94/tensorrt_llm/runtime/generation.py#L46
    flat_ids = []
    offsets = []
    for batch_stop_words in stop_words_list:
        item_flat_ids = []
        item_offsets = []

        for word in batch_stop_words:
            # there is a known issue in TensorRT-LLM that word ids are not unique and might change depending on
            # where in the text it appears. In our case we mainly need to stop on ids as they appear in the middle
            # of the text. The following is a workaround to get such ids that works for both <TOKEN> kind of stop
            # words as well as newlines that we commonly use. But note that it's not a universal fix, so this might
            # require refactoring if different stop words are used in the future.
            # Eventually, this needs to be fixed inside TensorRT-LLM itself.
            ids = tokenizer.encode('magic' + word)
            ids = ids[2:]  # skipping "magic"

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    stop_words = np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
    return torch.Tensor(stop_words).to(torch.int32).to("cuda").contiguous()


def load_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id



class TensorRTLLM:
    def __init__(self, model_path: str):
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(tokenizer_dir=model_path)
        self.runner = ModelRunnerCpp.from_dir(engine_dir=model_path, rank=tensorrt_llm.mpi_rank())
        self.runner.max_seq_len = self.runner.max_input_len
        
    @torch.no_grad()
    def forward(
        self,
        input_texts,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        batch_input_ids, input_lengths = parse_input(input_texts, self.tokenizer)
        if len(stop_words_list) > 0:
            stop_words_list = [stop_words_list for _ in range(len(input_texts))]
            stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, self.tokenizer)
            stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
        else:
            stop_words_list = None
        
        if max_output_token + max(input_lengths) > self.runner.max_seq_len:
            logging.warning(f"Set output token size from {max_output_token} to {self.runner.max_seq_len - max(input_lengths)}")
            max_output_token = min(max_output_token,  self.runner.max_seq_len - max(input_lengths))
            
            
        # TODO: return dictionary with a proper error reporting
        try:
            output_ids = self.runner.generate(
                batch_input_ids,
                max_new_tokens=max_output_token,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_words_list=stop_words_list,
                return_dict=False,
            )
            torch.cuda.synchronize()

            output = get_output(output_ids, input_lengths, max_output_token, self.tokenizer, self.end_id)
        except RuntimeError as e:
            logging.error("RuntimeError: %s", e)
            output = [f"RuntimeError: {e}"] * len(input_texts)

        return output


class WrapperServer:
    def __init__(self, model_path: str):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.model = TensorRTLLM(model_path=model_path)

        if self.rank == 0:
            self.app = Flask(__file__, static_url_path="")
            api = Api(self.app)
            api.add_resource(TritonServerGenerate, "/v1/completions", resource_class_args=[self.model])

    def run(self, url, port=5000):
        if self.rank == 0:
            self.app.run(url, threaded=True, port=port, debug=False)
        else:
            self.worker_loop()

    def worker_loop(self):
        triton = TritonServerGenerate(self.model)
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            triton.generate(**data)


if __name__ == "__main__":
    # TODO: can we reuse normal logger here?
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Start server...')
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    server = WrapperServer(model_path=args.model_path)
    server.run(args.host, args.port)
