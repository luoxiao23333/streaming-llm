###
 # @Author: Xiao Luo lxiao70@gatech.edu
 # @Date: 2023-10-13 22:32:48
 # @LastEditors: Xiao Luo lxiao70@gatech.edu
 # @LastEditTime: 2023-10-16 18:37:30
 # @FilePath: /streaming-llm/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# meta-llama/Llama-2-70b-chat-hf
# lmsys/vicuna-33b-v1.3
# lmsys/vicuna-13b-v1.3
# lmsys/vicuna-7b-v1.3
TRANSFORMERS_CACHE=tmp CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python examples/run_streaming_llama.py \
  --enable_streaming --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
  --cache_method StreamLLM --recent_size 1000

TRANSFORMERS_CACHE=tmp CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python examples/run_streaming_llama.py \
  --enable_streaming --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
  --cache_method StreamLLM --recent_size 1500

TRANSFORMERS_CACHE=tmp CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python examples/run_streaming_llama.py \
  --enable_streaming --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
  --cache_method StreamLLM --recent_size 2000

TRANSFORMERS_CACHE=tmp CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python examples/run_streaming_llama.py \
  --enable_streaming --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
  --cache_method NotCut

