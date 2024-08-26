import os
import argparse
import glob
import os
import subprocess

parser = argparse.ArgumentParser(description='scan_multi_nodes',
                                    allow_abbrev=False)

parser.add_argument('--master_node', action='store_true', default=False,
                    help='Master node or Slave node')
parser.add_argument('--num_nodes', type=int, default=4,
                    help='number of nodes')
parser.add_argument('--node_rank', type=int, default=0,
                    help='Index/Rank of the current node')
parser.add_argument('--ip_list', type=list, default=['10.11.8.151', '10.11.8.143', \
                                                     '10.11.8.152', '10.11.8.153', \
                                                     '10.11.8.142', '10.11.8.144', \
                                                     '10.11.8.145', '10.11.8.146', ],
                    help='the IP address for each nodes')
parser.add_argument('--master_port', type=int, default=23731,
                    help='Port number for Node2Node communication')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs')
parser.add_argument('--num_iters', type=int, default=20,
                    help='Number of training iterations per run')
parser.add_argument('--torch_compile', default=False,
                    action="store_true", help='using torch compile')
parser.add_argument('--torch_profile', default=False,
                    action="store_true", help='using torch profile')
parser.add_argument('--sync_files', default=False,
                    action="store_true", help='synchronize this file among different nodes')
parser.add_argument('--main_file', type=str, default='tune_basetrain.sh',
                    help='the main script for profiling')
parser.add_argument('--user_name', type=str, default='amd',
                    help='user name for remote nodes')
parser.add_argument('--repo_path', type=str, default=None, #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
args = parser.parse_args()

perf_test=[
    #mbs, gbs, tp, pp, profiling
    [1, 1,  8, 1, 0],
    [1, 2,  8, 1, 0],
    [1, 4,  8, 1, 0],
    [1, 8,  8, 1, 0],

    [2, 2,  8, 1, 0],
    [2, 4,  8, 1, 0],
    [2, 8,  8, 1, 0],
    [2, 16, 8, 1, 0],

    [3, 3,  8, 1, 0],
    [3, 6,  8, 1, 0],
    [3, 12, 8, 1, 0],
    [3, 24, 8, 1, 0],

    [4, 4,  8, 1, 0],
    [4, 8,  8, 1, 0], 
    [4, 16, 8, 1, 0],
    [4, 32, 8, 1, 0],

    [5, 5,  8, 1, 0],
    [5, 10, 8, 1, 0],
    [5, 20, 8, 1, 0],
    [5, 40, 8, 1, 0],

    [6, 6,  8, 1, 0],
    [6, 12, 8, 1, 0],
    [6, 24, 8, 1, 0],
    [6, 48, 8, 1, 0],
]
correct_test=[
    [1, 1,  8, 1, 1],
    [1, 2,  8, 1, 1],
    [2, 2,  8, 1, 1],
    [2, 4,  8, 1, 1],
    # [3, 3,  8, 1, 1],
    # [4, 4,  8, 1, 1],
#     [5, 5,  8, 1],
#     [6, 6,  8, 1],
]

correct_compile_test=[
    [1, 1, 8, 1, 1],
    [2, 2, 8, 1, 1],
    [3, 3, 8, 1, 1],
    [4, 4, 8, 1, 1],
    [5, 5, 8, 1, 1],
    [6, 6, 8, 1, 1],
]

configs_grad_overlap_comm_four_nodes = [
    [4, 32, 8, 1, 0],

    [5, 5,  8, 1, 0],
    [5, 40, 8, 1, 0],

    [6, 6,  8, 1, 0],
    [6, 48, 8, 1, 0],
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

def master_node():
    os.system('rm try_two_nodes_*')
    os.system('sleep 20')

    MASTER_IP = args.ip_list[0]
    for idx, config in enumerate(configs_grad_overlap_comm_four_nodes):
        mbs, gbs, tp, pp, profile_tag = config
        #'Micro batch,Global Batch,TP,PP'
        
        
        for node in range(0, args.num_nodes):
            file_name = 'try_two_nodes_{}_{}_{}.sh'.format(args.num_nodes, idx, node)
            gbs_scale = args.num_nodes * 8 //(tp*pp)
            configs = ' TP={} PP={} MBS={} BS={} NODE_RANK={} NNODES={} MASTER_PORT={} MASTER_ADDR={} TOTAL_ITERS={}'.format(tp, pp, mbs, gbs*gbs_scale, node, args.num_nodes, args.master_port, args.ip_list[0], args.num_iters)
            if args.torch_compile:
                configs=configs+ ' NO_TORCH_COMPILE=0'
            if args.torch_profile or profile_tag:
                configs=configs+ ' ENABLE_PROFILING=1'
            print('bash {}'.format(args.main_file)+configs, file=open(file_name,'w+'))
            if node>0:
                os.system('scp {} {}@{}:{}'.format(file_name, args.user_name, args.ip_list[node], args.repo_path))
        file_name = 'try_two_nodes_{}_{}_{}.sh'.format(args.num_nodes, idx, 0)
        os.system('bash {}'.format(file_name))
        
        while not check_all_ack(idx):
            os.system('sleep 1')
        os.system('sleep 10')
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
        os.system('scp {} {}@{}:{}'.format(ack_file_name, args.user_name, args.ip_list[0], args.repo_path))
    os.system('rm try_two_nodes_*')
    return None


def sync_file_among_nodes():
    assert args.node_rank==0
    all_lines = open('scan_multinode.py').readlines()
    for node_rank in range(1, args.num_nodes):
        os.system('rsync -av *.sh {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        os.system('rsync -av *.py {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        os.system('rsync -av megatron {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        # os.system('rsync -av pytorch_afo_testkit {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        os.system('rsync -av tasks {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        os.system('rsync -av tests {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))
        os.system('rsync -av tools {}@{}:{}/'.format(args.user_name, args.ip_list[node_rank], args.repo_path))

        out_file = 'sync_files_{}.py'.format(node_rank)
        with open(out_file, 'w+') as file_handler:
            for line_idx, line in enumerate(all_lines):
                line = line.replace('\n', '')
                if line_idx<20 and '--node_rank' in line and 'parser.add_argument(' in line:
                    line = 'parser.add_argument(\'--node_rank\', type=int, default={},'.format(node_rank)
                print(line, file=file_handler)
        os.system('scp {} {}@{}:{}/scan_multinode.py'.format(out_file, args.user_name, args.ip_list[node_rank], args.repo_path))

    os.system('rm sync_files_*')

def main():
    if args.repo_path is None:
        args.repo_path=subprocess.check_output(['pwd']).decode('utf-8')[:-1]

    if args.sync_files:
        sync_file_among_nodes()
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
