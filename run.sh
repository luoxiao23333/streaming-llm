###
 # @Author: Xiao Luo lxiao70@gatech.edu
 # @Date: 2023-10-13 22:32:48
 # @LastEditors: Xiao Luo lxiao70@gatech.edu
 # @LastEditTime: 2023-10-14 03:31:03
 # @FilePath: /streaming-llm/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming --model_name_or_path lmsys/longchat-7b-v1.5-32k
