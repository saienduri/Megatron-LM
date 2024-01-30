import sys
import re
import torch
import operator
from os import listdir
from os.path import isfile, join

kernels = {}

sum2 = 0

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    return is_rocm_pytorch

IS_CUDA = not check_if_rocm_pytorch()

def parse_kernel(name, group):
    global sum2

    if IS_CUDA:
        duration = int(group[1])
    else:
        duration = int(group[-2]) - int(group[-3])
    if duration < 0:
        #print(f'{name} \'s duration is {duration}')
        return 0
    if name in kernels:
        kernels[name] += duration
    else:
        kernels[name] = duration

    matched = re.search('nccl', name, re.IGNORECASE)
    #matched = re.search('at::', name)
    #matched = re.search('direct_copy_kernel_cuda', name)
    #matched = re.search('ck::', name)
    #matched = re.search('cutlass::', name)
    #matched = re.search('Cijk', name)
    #matched = re.search('_cublas', name)

    if matched:
        sum2 += duration

    return duration

def parse_kernels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        file_total = 0
        for i, line in enumerate(lines):
            if IS_CUDA:
                if i == 0:
                    continue
                if '\"' in line:
                    group = line.split('\"')
                    name = group[1]
                    file_total += parse_kernel(name, group[0].split(','))
                else:
                    group = line.split(',')
                    name = group[-1]
                    file_total += parse_kernel(name, group[:-1])
            else:
                if i == 0 or i == 1:
                    continue
                group = line.split('\"')
                name = group[1]
                file_total += parse_kernel(name, group[-1].split(','))

        return file_total

def main():
    total = 0
    if IS_CUDA:
        total = parse_kernels(sys.argv[1])
    else:
        mydir = sys.argv[1]
        files = [f for f in listdir(mydir) if isfile(join(mydir, f)) and 'result' in f]
        for the_csv in files:
            total += parse_kernels(join(mydir, the_csv))

    kernels_sorted = dict(sorted(kernels.items(), key=operator.itemgetter(1), reverse=True))

    #print('sum 1 =', sum([v for v in kernels_sorted.values()]))
    #print('sum 2 =', sum2)

    for k in kernels_sorted:
        percentage = float(kernels_sorted[k])/total
        totaltime = float(kernels_sorted[k])
        print(f"{k},{percentage:.1%},{totaltime}".format())

if __name__ == '__main__':
    main()
