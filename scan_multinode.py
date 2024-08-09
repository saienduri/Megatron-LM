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
parser.add_argument('--ip_list', type=list, default=['10.11.8.150', '10.11.8.151', '10.11.8.152','10.11.8.140'],
                    help='Set number of training iterations for cyclic '
                    'Retro training.')
parser.add_argument('--master_port', type=int, default=23731,
                    help='Number of layers to use for the retrieval '
                    'encoder.')
parser.add_argument('--node_rank', type=int, default=0,
                    help='Number of layers to use for the retrieval '
                    'encoder.')
args = parser.parse_args()

def convert_tempalte(TP, PP, rank):
    template={
        29: 'MASTER_ADDR={}'.format(args.ip_list[0]),
        30: 'MASTER_PORT={}'.format(args.master_port),
        31: 'NNODES={}'.format(args.num_nodes),
        32: 'NODE_RANK={}'.format(rank),
        36: 'TP=\"${' +'TP:-'+str(TP)+'}\"', #'TP=\"\$\{TP\:-{}\}\"'.format(TP),
        37: 'PP=\"${' +'PP:-'+str(PP)+'}\"' #'PP=\"$\{PP\:-{}\}\"'.format(PP),
    }
    alllines = open('basetrain70b.sh', 'r')
    file_name = 'multinode_tp_{}_pp_{}_rank_{}.sh'.format(TP, PP, rank)
    with open(file_name, 'w+') as outfile:
        for lineid, line in enumerate(alllines):
            if lineid in template:
                print(template[lineid], file=outfile)
            else:
                print(line.replace('\n', ''), file=outfile)
    return file_name


def master_node():
    MASTER_IP = args.ip_list[0]
    for TP in [8]:
        for PP in [4]:
            if TP*PP<=8*args.num_nodes:
                for node in range(args.num_nodes):
                    file_name = convert_tempalte(TP, PP, node)
                    if node>0:
                        os.system('scp {} amd@{}:/home/amd/guihong/megatron-lm-jiang'.format(file_name, args.ip_list[node]))
                        os.system('rm {}'.format(file_name))

        # os.system('bash test.sh')
    return None


def slave_node():
    search_dir = "./"
    # remove anything from the list that is not a file (directories, symlinks)
    # thanks to J.F. Sebastion for pointing out that the requirement was a list 
    # of files (presumably not including directories)  
    files = list(filter(os.path.isfile, glob.glob(search_dir + "multinode_*_rank_{}.sh".format(args.node_rank))))
    files.sort(key=lambda x: os.path.getmtime(x))
    print(files)

    return None


if args.master_node:
    master_node()
    slave_node()
else:
    slave_node()
