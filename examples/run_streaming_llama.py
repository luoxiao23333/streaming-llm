import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

#os.environ["TRANSFORMERS_CACHE"] = "tmp"

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm import utils
from streaming_llm.kv_cache import StartRecentKVCache

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, out=sys.stdout):
    start_t = time.time()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    mem_consumption = utils.get_gpu_mem_allocated()
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            mem_consumption = max(mem_consumption, utils.get_gpu_mem_allocated())
            with open("./memory.txt",  "+a") as file:   
                file.write(f"Memory is {utils.get_gpu_mem_allocated()} GB\n")
            print(" ".join(generated_text[pos:now]), end=" ", flush=True, file=out)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True, file=out)
    return past_key_values, mem_consumption, (time.time() - start_t) / len(generated_text)


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, cache_method, kv_cache=None, max_gen_len=1000, out=sys.stdout):
    past_key_values = None
    results = []
    for idx, prompt in enumerate(prompts):
        if(idx == 2):
            break
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="", file=out)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            # remain space for encoding{seq_len} and KV in future self-regressive{max_gen_len}
            space_needed = seq_len + max_gen_len
            # recent window = [past_seq_len-recent_size+input_seq_len+max_gen_len, seq_len]
            past_key_values = kv_cache.get_method(cache_method)(past_key_values, space_needed)

        cache_mem = 0 if past_key_values is None else (len(past_key_values)*len(past_key_values[0])*past_key_values[0][0].element_size()*past_key_values[0][0].numel()) / (1024 ** 3)
        # mem = "None" if past_key_values is None else past_key_values[0][0].shape
        past_key_values, mem_allocated, latency = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, out=out
        )
        results.append({"mem": mem_allocated, "latency": latency, "cache_mem": cache_mem})
    return results


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    print(model)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    out = open("output.txt", "a+")
    out.write(f"With Model: {args.model_name_or_path}, With Method: {args.cache_method}\n")

    results = streaming_inference(
        model,
        tokenizer,
        prompts,
        args.cache_method,
        kv_cache,
        out=out
    )

    out.write("\n\n{}\n\n".format("-"*60))

    with open("result.txt", "a+") as file:
        file.write(f"With Model {args.model_name_or_path}, With Method: {args.cache_method}\n")
        for idx, result in enumerate(results):
            file.write("Seq: {}, Mem: {:.4f} GB, KV Cached: {:.4f} GB, Latency: {:.4f} seconds/word\n".format(idx, result["mem"], result["cache_mem"], result["latency"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--cache_method", type=str, choices=StartRecentKVCache.get_supported_method())
    args = parser.parse_args()

    main(args)
