import sys
import re
import csv

pattern_sps = r'samples per second: (\d+\.\d+)'

pattern_gbs = '--global-batch-size (\d+)'
pattern_mbs = '--micro-batch-size (\d+)'
pattern_tp = '--tensor-model-parallel-size (\d+)'
pattern_pp = '--pipeline-model-parallel-size (\d+)'

def get_string(line, pattern):
    matched = re.search(pattern, line)
    if matched:
        return matched.group(1)
    else:
        return None

def get_strings(lines):
    count = 0
    total = 0
    for line in lines:
        ret = get_string(line, pattern_sps)
        if ret:
            count += 1
            total += float(ret)
        ret = get_string(line, pattern_gbs)
        if ret:
            gbs = int(ret)
        ret = get_string(line, pattern_mbs)
        if ret:
            mbs = int(ret)
        ret = get_string(line, pattern_tp)
        if ret:
            tp = int(ret)
        ret = get_string(line, pattern_pp)
        if ret:
            pp = int(ret)

    sps = total/count
    return [gbs, mbs, tp, pp, sps]

with open(sys.argv[1], 'r') as f_in:
    row = get_strings(f_in.readlines())
    print(row, 'written')

    with open('output.csv', 'a') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(row)
