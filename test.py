'''
Author: Xiao Luo lxiao70@gatech.edu
Date: 2023-10-15 13:38:33
LastEditors: Xiao Luo lxiao70@gatech.edu
LastEditTime: 2023-10-15 16:54:49
FilePath: /streaming-llm/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import subprocess

# 定义 cache_method 和 model_name_or_path 的组合
cache_methods = ["NotCut", "StreamLLM", "SparseSample"]
model_name_or_paths = [
    "lmsys/longchat-7b-v1.5-32k",
    "lmsys/longchat-13b-16k",
    "lmsys/vicuna-7b-v1.3",
    "lmsys/vicuna-13b-v1.3",
    "lmsys/vicuna-33b-v1.3",
    "meta-llama/Llama-2-70b-chat-hf",
]


# 循环遍历组合执行命令


for model_name_or_path in model_name_or_paths:
    for cache_method in cache_methods:
        command = f"TRANSFORMERS_CACHE=tmp CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/run_streaming_llama.py --enable_streaming --model_name_or_path {model_name_or_path} --cache_method {cache_method}"
        print(f"Start to execute {command}")
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Command executed successfully: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}")
            print(e)
