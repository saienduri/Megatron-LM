import sys
import re

pattern_at = r'at::'
pattern_cublas = '_cublas'
pattern_cutlass = 'cutlass'

# Step 1: HIP_VISIBLE_DEVCIES=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=1024 MICRO_BATCH_SIZE=6 TP_SIZE=1 PP_SIZE=1 nsys profile -o megatron examples_deepspeed/pretrain_gpt_with_mp.sh
# Step 2: nsys stats --report gpukernsum --format csv megatron.nsys-rep

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    sum = 0
    for line in lines:
        matched = re.search(pattern_cutlass, line)
        #matched = re.search(pattern_at, line)
        #matched = re.search(pattern_cublas, line)
        if matched:
            #print('found!')
            sum += float(line.split(',')[0])

    print(sum)
