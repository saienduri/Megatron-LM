import argparse
import subprocess
import concurrent.futures

# Configuration: List of servers (assuming SSH keys are set up for authentication)

# 16 nodes:

all_servers = ["tw042", "tw044", "tw045", "tw046", "tw043", "tw053", "tw052", "tw051"]

parser = argparse.ArgumentParser(description='launch the training on multi nodes',
                                    allow_abbrev=False)
parser.add_argument('--num_nodes', type=int, default=8,
                    help='number of nodes')
parser.add_argument('--task_id', type=int, default=1,
                    help='0: stop jobs, 1: training, 2: check gpu utilization')
parser.add_argument('--repo_path', type=str, default=None, #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--repo_path_upper', type=str, default='/home/amd/guihong', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--home_path', type=str, default='/home/amd', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
parser.add_argument('--docker_image_name', type=str, 
                    default='rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_TE', #'~/guihong/megatron_lm',
                    help='the path where storing this folder')
args = parser.parse_args()

node_ranks = {}
servers = all_servers[len(all_servers)-args.num_nodes:]
for r in range(args.num_nodes):
    node_ranks[servers[r]] = r

task_id = args.task_id
user_name = "gl" # Specify User Name
container_name = f"test_megatron_{user_name}"

tasks = {0: "stop jobs", 1: "training", 2: "check gpu utilization"}

multinodes_training = False
args.repo_path = subprocess.check_output(['pwd']).decode('utf-8')[:-1]
args.home_path = subprocess.check_output(['echo', '$HOME']).decode('utf-8')[:-1]

def generate_command():
    if args.task_id == 0:
        command = f"docker stop {container_name} && docker rm {container_name} exit"
        print("Stopping docker containers ...")
    elif args.task_id == 1:
        command = f"docker run  --device /dev/dri --device /dev/kfd \
            --network host --ipc host --group-add video \
            --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
            --privileged  -v {args.repo_path_upper}:{args.repo_path_upper}   \
            -v  {args.home_path}/.ssh:/root/.ssh  --shm-size 64G \
            --name {container_name}  {args.docker_image_name} \
            /bin/bash -c \"cd {args.repo_path} && source install.sh && python scan_multinode.py --torch_profile\""
        command = f"docker run  --device /dev/dri --device /dev/kfd \
            --network host --ipc host --group-add video \
            --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
            --privileged  -v {args.repo_path_upper}:{args.repo_path_upper}   \
            -v  {args.home_path}/.ssh:/root/.ssh  --shm-size 64G \
            --name {container_name}  {args.docker_image_name} \
            /bin/bash -c \"cd {args.repo_path} && source install.sh && bash tune_basetrain.sh\""
        multinodes_training = True
        print("Starting Multi-nodes Training ...")
    elif args.task_id == 2:
        command = "rocm-smi; exit"
        print("Showing gpu utilization ...")
    else:
        raise NotImplementedError("Undefiend Task")
    return command
# command = "docker ps -a; exit"
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