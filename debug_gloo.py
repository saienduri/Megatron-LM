import os
import torch
import torch.distributed as dist

def print_specific_envs():
    env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f'{var}: {value}')
        
def init_distributed_mode():
    print_specific_envs()  # Print specific environment variables for debugging

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(
        backend='gloo',  # Use 'gloo' for CPU communication
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    init_distributed_mode()