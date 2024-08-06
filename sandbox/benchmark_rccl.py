import torch
import torch.distributed as dist
import os
from torch.profiler import profile, record_function, ProfilerActivity

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def run(rank, size, dtype=torch.bfloat16, M=8192, N=8192, K=8192):
    """ Distributed function to perform all-gather and reduce-scatter. """
    tensor = rank * torch.ones((4096, 2, 8192), dtype=dtype, device=f'cuda:{rank}')
    a = torch.randn((M, K), dtype=dtype, device=f'cuda:{rank}')
    b = torch.randn((K, N), dtype=dtype, device=f'cuda:{rank}')
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(size)]
    output_tensor = torch.zeros_like(tensor)

    def fn(tensor, a, b, gathered_tensors=gathered_tensors, output_tensor=output_tensor):
        # Perform all-gather
        b.matmul(a)
        dist.all_gather(gathered_tensors, tensor)
        c = b.matmul(a)
        
        # Concatenate gathered tensors into one tensor
        gathered_tensors = torch.cat(gathered_tensors, dim=0)

        b.matmul(a)
        # Reduce-scatter the concatenated tensor
        dist.reduce_scatter(output_tensor, list(gathered_tensors.chunk(size, dim=0)))
        d = b.matmul(a)
        
        print(f"Rank {rank} finished with tensor shape: {output_tensor.shape}")


        return output_tensor, c, d

    for i in range(4):
        fn(tensor, a, b)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                with_modules=True,
                profile_memory=True,
                with_stack=True) as prof:
        fn(tensor, a, b)
    # print(prof.key_averages().table(row_limit=10))
    fname_prefix = f"run_rank{torch.distributed.get_rank()}"
    prof.export_chrome_trace(f"{fname_prefix}-trace.json")


if __name__ == "__main__":
    size = 8
    processes = []
    
    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
