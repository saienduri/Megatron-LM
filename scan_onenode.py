import os
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='scan_multi_nodes',
                                    allow_abbrev=False)

parser.add_argument('--master_node', action='store_true', default=False,
                    help='Master node or Slave node')
parser.add_argument('--num_nodes', type=int, default=4,
                    help='number of nodes')
parser.add_argument('--node_rank', type=int, default=0,
                    help='Index/Rank of the current node')
parser.add_argument('--ip_list', type=list, default=['10.11.8.150', '10.11.8.151', '10.11.8.143', '10.11.8.152'],
                    help='the IP address for each nodes')
parser.add_argument('--master_port', type=int, default=23731,
                    help='Port number for Node2Node communication')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs')
parser.add_argument('--num_iters', type=int, default=6,
                    help='Number of training iterations per run')
parser.add_argument('--torch_compile', default=False,
                    action="store_true", help='using torch compile')
parser.add_argument('--torch_profile', default=False,
                    action="store_true", help='using torch profile')
parser.add_argument('--sync_such_nodes', default=False,
                    action="store_true", help='synchronize this file among different nodes')
parser.add_argument('--main_file', type=str, default='tune_basetrain.sh',
                    help='synchronize this file among different nodes')
args = parser.parse_args()

config_batchs = [
    #'Micro batch,Global Batch,TP,PP'
    [1,1,8,1],
    [1,1,4,2],
    [1,1,2,4],
    [1,1,1,8],
    [1,2,8,1],
    [1,2,4,2],
    [1,2,2,4],
    [1,2,1,8],
    [2,2,8,1],
    [2,2,4,2],
    [2,2,2,4],
    [2,2,1,8],
    [1,4,8,1],
    [1,4,4,2],
    [1,4,2,4],
    [1,4,1,8],
    [2,4,8,1],
    [2,4,4,2],
    [2,4,2,4],
    [2,4,1,8],
    [1,8,8,1],
    [1,8,4,2],
    [1,8,2,4],
    [1,8,1,8],
    [2,8,8,1],
    [2,8,4,2],
    [2,8,2,4],
    [2,8,1,8],
]

configs_grad_overlap_comm=[
    [1,1,8,1],
    [1,2,8,1],
    [1,4,8,1],
    [1,8,8,1],
    [2,2,8,1],
    [2,4,8,1],
    [2,8,8,1],
    [2,16,8,1],
    [4,4,8,1],
    [4,8,8,1],
    [4,16,8,1],
    [4,32,8,1],
    [5,5,8,1],
    [5,10,8,1],
    [5,20,8,1],
    [5,40,8,1],
]

#mbs, gbs, tp, pp
configs_grad_overlap_comm_four_nodes=[
    # [2, 2, 8, 1],
    # [4, 16, 8, 1],
    # [4, 4, 8, 1],
    # [5, 5, 8, 1],
    [4, 16, 8, 1], 
    [4, 16, 8, 1], 
    [4, 16, 8, 1], 
    [4, 16, 8, 1], 
    # # [4,4,  8,1],
    # [4,64, 8,1],
    # [5,5,  8,1],
    # # [5,80, 8,1],
    # [6,6,  8,1],
    # # [6,96, 8,1],
    # # [4,16, 8,1],
    # # [4,32, 8,1],
    # # [4,8,  8,1],
]

configs_grad_overlap_comm_gemm=[
    [5,5,8,1],
]

def raw_scan():
    for mbs in [2,1]:
        for gbs_scale in [1,2,4,8]:
            for total_parallel in [8]:
                tp = total_parallel
                minimal_gbs = 8//total_parallel
                while tp>4:
                    pp = total_parallel//tp
                    gbs = minimal_gbs*gbs_scale*mbs*args.num_nodes
                    configs = 'bash {} TP={} PP={} MBS={} BS={}'.format(args.main_file, tp, pp, mbs, gbs)
                    if args.torch_compile:
                        configs=configs+ ' NO_TORCH_COMPILE=0'
                    if args.torch_profile:
                        configs=configs+ ' ENABLE_PROFILING=1'
                    os.system(configs)
                    print(configs)
                    tp = tp//2
                    os.system('sleep 10')

def check_all_ack(idx):
    got_all_file=True
    for i in range(1, args.num_nodes):
        ack_file_name = 'try_two_nodes_{}_{}_{}_ack.sh'.format(args.num_nodes, idx, i)
        got_all_file = got_all_file and os.path.isfile(ack_file_name)
    return got_all_file
'''
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
ENABLE_PROFILING="${ENABLE_PROFILING:-1}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
'''
def master_node():
    os.system('rm try_two_nodes_*')
    os.system('sleep 60')

    MASTER_IP = args.ip_list[0]
    for idx, config in enumerate(configs_grad_overlap_comm_four_nodes):
        mbs, gbs, tp, pp = config
        #'Micro batch,Global Batch,TP,PP'
        
        
        for node in range(0, args.num_nodes):
            file_name = 'try_two_nodes_{}_{}_{}.sh'.format(args.num_nodes, idx, node)
            gbs_scale = args.num_nodes * 8 //(tp*pp)
            configs = ' TP={} PP={} MBS={} BS={} NODE_RANK={} NNODES={} MASTER_PORT={} MASTER_ADDR={} TOTAL_ITERS={}'.format(tp, pp, mbs, gbs*gbs_scale, node, args.num_nodes, args.master_port, args.ip_list[0], args.num_iters)
            if args.torch_compile:
                configs=configs+ ' NO_TORCH_COMPILE=0'
            if args.torch_profile:
                configs=configs+ ' ENABLE_PROFILING=1'
            print('bash {}'.format(args.main_file)+configs, file=open(file_name,'w+'))
            if node>0:
                os.system('scp {} amd@{}:/home/amd/guihong/megatron-lm-jiang'.format(file_name, args.ip_list[node]))
        file_name = 'try_two_nodes_{}_{}_{}.sh'.format(args.num_nodes, idx, 0)
        os.system('bash {}'.format(file_name))
        
        while not check_all_ack(idx):
            os.system('sleep 1')
        os.system('sleep 60')
        # os.system('bash test.sh')
    os.system('rm try_two_nodes_*')
    return None


def slave_node():
    os.system('rm try_two_nodes_*')
    for idx, config in enumerate(configs_grad_overlap_comm_four_nodes):
        file_name = 'try_two_nodes_{}_{}_{}.sh'.format(args.num_nodes, idx, args.node_rank)
        while not os.path.isfile(file_name):
            os.system('sleep 1')
        os.system('bash {}'.format(file_name))
        ack_file_name = 'try_two_nodes_{}_{}_{}_ack.sh'.format(args.num_nodes, idx, args.node_rank)
        print('DONE!', file=open(ack_file_name,'w+'))
        os.system('scp {} amd@{}:/home/amd/guihong/megatron-lm-jiang'.format(ack_file_name, args.ip_list[0]))
    os.system('rm try_two_nodes_*')
    return None

def main():
    if args.sync_such_nodes:
        assert args.node_rank==0
        all_lines = open('scan_onenode.py').readlines()
        for node_rank in range(1, args.num_nodes):
            os.system('scp *.sh amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format( args.ip_list[node_rank]))
            os.system('scp *.py amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format( args.ip_list[node_rank]))
            # os.system('scp basetrain70b.sh amd@{}:/home/amd/guihong/megatron-lm-jiang/basetrain70b.sh'.format(args.ip_list[node_rank]))
            # os.system('scp pretrain_gpt.py amd@{}:/home/amd/guihong/megatron-lm-jiang/pretrain_gpt.py'.format(args.ip_list[node_rank]))
            os.system('rsync -av megatron amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))
            os.system('rsync -av pytorch_afo_testkit amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))
            os.system('rsync -av pytorch_afo_testkit amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))
            os.system('rsync -av tasks amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))
            os.system('rsync -av tests amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))
            os.system('rsync -av tools amd@{}:/home/amd/guihong/megatron-lm-jiang/'.format(args.ip_list[node_rank]))

            out_file = 'sync_such_nodes_{}.py'.format(node_rank)
            with open(out_file, 'w+') as file_handler:
                for line_idx, line in enumerate(all_lines):
                    line = line.replace('\n', '')
                    if line_idx<20 and '--node_rank' in line and 'parser.add_argument(' in line:
                        line = 'parser.add_argument(\'--node_rank\', type=int, default={},'.format(node_rank)
                    print(line, file=file_handler)
            os.system('scp {} amd@{}:/home/amd/guihong/megatron-lm-jiang/scan_onenode.py'.format(out_file, args.ip_list[node_rank]))

        os.system('rm sync_such_nodes_*')

    else:
        for i in range(args.num_runs):
            if args.num_nodes==1:
                raw_scan()

            elif args.num_nodes>1:
                if args.master_node or args.node_rank==0:
                    master_node()
                else:
                    slave_node()

if __name__ == "__main__":
    main()
