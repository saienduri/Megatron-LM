import argparse
import subprocess
import concurrent.futures
import os

# Configuration: List of servers (assuming SSH keys are set up for authentication)

# 16 nodes:

all_servers = ["tw042", "tw044", "tw045", "tw046", "tw043", "tw053", "tw052", "tw051"]
# all_servers = ["tw025", "tw026", "tw027", "tw028", "tw033", "tw034", "tw037", "tw038"]

all_servers = ['tw042', 'tw044', 'tw045', 'tw052', 'tw025', 'tw026', 'tw028', 'tw033']
# all_servers = ["tw043", "tw046", "tw051", "tw053", "tw046", "tw037", "tw038", "tw027"]
all_servers = ['tw042', 'tw044', 'tw045', 'tw052', 'tw025', 'tw026', 'tw028', 'tw033']

parser = argparse.ArgumentParser(description='launch the training on multi nodes',
                                    allow_abbrev=False)
parser.add_argument('--num_nodes', type=int, default=8,
                    help='number of nodes')
parser.add_argument('--task', type=str, default=None, required=True,
                    choices=['kill-docker', 'make-sh', 'set-ib', 'launch-docker', 'check-ib', 'sync-file', 'check-docker', 'run', 'amd-smi', 'rocm-smi', 'kill-all'],
                    help='kill: stop jobs, run: training, gpu-utils: check gpu utilization')
parser.add_argument('--repo_path', type=str, default=None, #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--repo_path_upper', type=str, default='/home/amd/guihong', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--home_path', type=str, default='/home/amd', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--docker_image_name', type=str, 
                    default='rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_bf16_rtn_TE_GQA', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--master_port', type=int, default=37876,
                    help='Port number for Node2Node communication')
parser.add_argument('--seq_length', type=int, default=128000,
                    help='Number of training iterations per run')
parser.add_argument('--main_file', type=str, default='train_acc_loss_llama3.sh',
                    help='the main script for profiling')
parser.add_argument('--num_iters', type=int, default=10,
                    help='Number of training iterations per run')
parser.add_argument('--torch_profile', default=False,
                    action="store_true", help='using torch profile')
parser.add_argument('--user_name', type=str, default='amd',
                    help='user name for remote nodes')
args = parser.parse_args()

args.docker_image_name = 'rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_bf16_rtn_TE_GQA'
# args.docker_image_name = 'rocm/pytorch-private:pytorch25_te_1_11_fa_2_py_39_guihong'

node_ranks = {}
servers = all_servers[len(all_servers)-args.num_nodes:]
for r in range(args.num_nodes):
    node_ranks[servers[r]] = r

task = args.task
user_name = "gl" # Specify User Name
container_name = f"test_megatron_{user_name}"
container_name = f"te_test_cp_gl"

tasks = {0: "stop jobs", 1: "training", 2: "check gpu utilization"}

multinodes_training = False
args.repo_path = subprocess.check_output(['pwd']).decode('utf-8')[:-1]
args.home_path = subprocess.check_output(['echo', '$HOME']).decode('utf-8')[:-1]


configs_grad_overlap_comm_four_nodes = [
    # model_size, mbs, grad_accumsteps, tp, pp, cp, profiling, compiling
    # [70,  1,  1,  8,  2,  2, 0,  0],
    [8,   1,  4,  8,  1,  1, 0,  0],
    # [70,  1,  4,  8,  1,  4, 0,  0],
]

def sync_file_among_nodes():

    for node_rank in range(1, args.num_nodes):
        os.system('rsync -av *.sh {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        os.system('rsync -av *.py {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        os.system('rsync -av megatron {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        # os.system('rsync -av pytorch_afo_testkit {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        os.system('rsync -av tasks {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        os.system('rsync -av tests {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))
        os.system('rsync -av tools {}@{}:{}/'.format(args.user_name, all_servers[node_rank], args.repo_path))


def make_shell():
    MASTER_IP = servers[0]
    for node in list(range(0, args.num_nodes))[::-1]:
        file_name = 'tmp_run_multinode_tmp.sh'.format()
        with open(file_name,'w+') as file_handler:
            for idx, config in enumerate(configs_grad_overlap_comm_four_nodes):
                modelsize, mbs, accum_steps, tp, pp, cp, profile_tag, compiling_tag = config

                gbs = mbs*accum_steps*args.num_nodes * 8 //(tp*pp*cp)
                configs = ' TP={} PP={} CP={} MBS={} BS={} NODE_RANK={} NNODES={} MASTER_PORT={} MASTER_ADDR={} TOTAL_ITERS={} SEQ_LENGTH={} '.format(\
                    tp, pp, cp, mbs, gbs, node, args.num_nodes, args.master_port, servers[0], args.num_iters, args.seq_length)
                configs = configs+' MODEL_SIZE={}'.format(modelsize)
                # if args.torch_compile or compiling_tag:
                #     configs=configs+ ' NO_TORCH_COMPILE=0'
                if args.torch_profile or profile_tag:
                    configs=configs+ ' ENABLE_PROFILING=1'
                print('bash {} '.format(args.main_file)+configs+'\n\n', file=file_handler)
            
        os.system('scp {} {}@{}:{}/tmp_run_multinode_tmp.sh'.format(file_name, args.user_name, servers[node], args.repo_path))


def generate_command():
    if args.task == 'kill-docker':
        command = f"docker stop {container_name} && docker rm {container_name} exit"
        print("Stopping docker containers ...")
    elif args.task == 'launch-docker':
        command = f"docker run  --device /dev/dri --device /dev/kfd \
            --network host --ipc host --group-add video \
            --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
            --privileged  -v {args.repo_path_upper}:{args.repo_path_upper}   \
            -v  {args.home_path}/.ssh:/root/.ssh  --shm-size 64G \
            --name {container_name}  {args.docker_image_name}"
    elif args.task == 'check-docker':
        command = f"docker images -a  | grep  {args.docker_image_name}"
    elif args.task == 'set-ib':
        command = f"docker exec {container_name} bash -c 'cd {args.repo_path}/../set_bc_ib && bash set_bc_ib.sh' "
    elif args.task == 'check-ib':
        command = f"docker exec {container_name} bash -c 'ibv_devices' "
    elif args.task == 'run':
        command = f"docker run  --device /dev/dri --device /dev/kfd \
            --network host --ipc host --group-add video \
            --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
            --privileged  -v {args.repo_path_upper}:{args.repo_path_upper}   \
            -v  {args.home_path}/.ssh:/root/.ssh  --shm-size 64G \
            --name {container_name}  {args.docker_image_name} \
            /bin/bash -c 'cd {args.repo_path} && source install.sh' "
        multinodes_training = True
        print("Starting Multi-nodes Training ...")
    elif args.task =="rocm-smi":
        command = "rocm-smi; exit"
        print("Showing gpu utilization ...")
    elif args.task =="amd-smi":
        command = "sudo amd-smi process; exit"
        print("Showing gpu processes ...")
    elif args.task =="kill-all":
        command=r"sudo amd-smi process | grep -oP 'PID: \K\d+' | xargs -r sudo kill -9"
    elif args.task =='make-sh' or args.task =='sync-file':
        command = None
        pass
    else:
        raise NotImplementedError("Undefiend Task")
    return command

command = generate_command()
print('\n')
timeout_t = 180

# Function to execute a command on a remote server
def execute_ssh_command(server):
    host = server
    print("Running on host : {} ".format(host))
    ssh_command = ["ssh", f"{host}"]
    exec_command = command
    ssh_command.extend(exec_command.split())
    try:
        # Run the SSH command and capture the output
        result = subprocess.run(ssh_command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_t)
        return f"\nOutput from {host}:\n{result.stdout}"
    # except subprocess.CalledProcessError as e:
    except Exception as e:
        # Handle errors in the subprocess
        return f"\nError executing command on {host}:\n{e}\n{e.stderr}"

# Use ThreadPoolExecutor to run SSH commands in parallel
def main():
    # Define the number of max workers, could be len(servers) if you want to max parallelism
    all_res = []
    if args.task=='make-sh':
        make_shell()
    elif args.task=='sync-file':
        sync_file_among_nodes()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Map the execute_ssh_command to the servers
            results = executor.map(execute_ssh_command, servers)


            # Output results as they are completed
            for result in results:
                # print(result)
                all_res.append(result)

        print("\n Final results ..... ")
        for result in all_res:
            print(result)

if __name__ == "__main__":

    main()