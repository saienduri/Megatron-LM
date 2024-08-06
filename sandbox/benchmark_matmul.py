
import torch
import torch.nn as nn
import triton
import triton.ops.matmul as triton_matmul


# llama2-13B, MBS=1
# 4096	128	4096
# 128	4096	4096
# 5120	15360	4096
# 5120	5120	4096
# 5120	27648	4096
# 13824	5120	4096
# 5120	32000	4096
# 5120	4096	15360
# 5120	4096	5120
# 5120	4096	27648
# 13824	4096	5120
# 5120	4096	32000
# 128	4096	4096
# 4096	4096	128
# 32000	4096	5120
# 27648	4096	5120
# 5120	4096	13824
# 5120	4096	5120
# 15360	4096	5120


# llama2-70B
# trnaspose	DTYPE	M	N	K	B
# NT	torch.bfloat16	8192	1280	4096	1
# NT	torch.bfloat16	1024	8192	4096	1
# NT	torch.bfloat16	8192	7168	4096	1
# NT	torch.bfloat16	3584	8192	4096	1
# NT	torch.bfloat16	8192	4096	4096	1
# NN	torch.bfloat16	8192	4096	1280	1
# NN	torch.bfloat16	1024	4096	8192	1
# NN	torch.bfloat16	8192	4096	7168	1
# NN	torch.bfloat16	3584	4096	8192	1
# NN	torch.bfloat16	8192	4096	4096	1
# TN	torch.bfloat16	4096	4096	128	8
# TN	torch.bfloat16	4096	4096	8192	1
# TN	torch.bfloat16	8192	4096	3584	1
# TN	torch.bfloat16	7168	4096	8192	1
# TN	torch.bfloat16	8192	4096	1024	1
# TN	torch.bfloat16	1280	4096	8192	1
# NT	torch.float16	5120	32000	4096	1
# NT	torch.float16	13824	5120	4096	1
# NT	torch.float16	5120	27648	4096	1
# NT	torch.float16	5120	5120	4096	1
# NN	torch.bfloat16	128	4096	4096	8
# NN	torch.float16	5120	4096	32000	1
# NN	torch.float16	13824	4096	5120	1
# NN	torch.float16	5120	4096	27648	1
# NN	torch.float16	5120	4096	5120	1
# NN	torch.float16	128	4096	4096	40
# TN	torch.float16	4096	4096	128	40
# NT	torch.bfloat16	4096	128	4096	8
# NT	torch.bfloat16	128	4096	4096	8
# NT	torch.float16	128	4096	4096	40
# TN	torch.float16	15360	4096	5120	1
# TN	torch.float16	5120	4096	5120	1
# TN	torch.float16	5120	4096	13824	1
# TN	torch.float16	27648	4096	5120	1
# TN	torch.float16	32000	4096	5120	1

global verbose
verbose = True

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K', 'B'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            ( 8192, 1280, 4096, 1),
            ( 1024, 8192, 4096, 1),
            ( 8192, 7168, 4096, 1),
            ( 3584, 8192, 4096, 1),
            ( 8192, 4096, 4096, 1),
            ( 8192, 4096, 1280, 1),
            ( 1024, 4096, 8192, 1),
            ( 8192, 4096, 7168, 1),
            ( 3584, 4096, 8192, 1),
            ( 8192, 4096, 4096, 1),
            # # (torch.bfloat16, 4096, 4096, 128, 8),
            ( 4096, 4096, 8192, 1),
            ( 8192, 4096, 3584, 1),
            ( 7168, 4096, 8192, 1),
            ( 8192, 4096, 1024, 1),
            ( 1280, 4096, 8192, 1),
            # ( 5120, 32000, 4096, 1),
            # ( 13824, 5120, 4096, 1),
            # ( 5120, 27648, 4096, 1),
            # ( 5120, 5120, 4096, 1),
            # # ( 128, 4096, 4096, 8),
            # ( 5120, 4096, 32000, 1),
            # ( 13824, 4096, 5120, 1),
            # ( 5120, 4096, 27648, 1),
            # ( 5120, 4096, 5120, 1),
            # # ( 128, 4096, 4096, 40),
            # # ( 4096, 4096, 128, 40),
            # # ( 4096, 128, 4096, 8),
            # # ( 128, 4096, 4096, 8),
            # # ( 128, 4096, 4096, 40),
            # ( 15360, 4096, 5120, 1),
            # ( 5120, 4096, 5120, 1),
            # ( 5120, 4096, 13824, 1),
            # ( 27648, 4096, 5120, 1),
            # ( 32000, 4096, 5120, 1),
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['matmul',  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        line_names=['matmul',  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        # line_vals=['matmul-a.t'],#,  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        # line_names=['matmul-a.t'],#,  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        # Line styles
        styles=[('k', '--'), ('k', '-'), ('red', '-'), ('red', '--'), ('g', '-'), ('g', '--'), ('b', '-'), ('b', '--'),
                ('m', '-'), ('m', '--'), ('c', '-'), ('y', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance.tune",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, B, provider, dtype=torch.bfloat16, quantiles = [0.5, 0.2, 0.8]):
    print(f"## {provider} benchmark {M}, {N}, {K} ##")
    print(dtype)
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    b = torch.randn((K, N), dtype=dtype, device='cuda')
    # b = torch.randn((N, K), device='cuda', dtype=dtype)
    # b = b.t()
    
    if provider == 'bmm':
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul':
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-a.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        a = a.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-b.t':
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-a.t-b.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        a = a.t()
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'triton' or provider == 'triton_matmul':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b), quantiles=quantiles)
        fn = lambda: triton_matmul(a, b)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    if verbose:
        print(f"latency: ms : {ms}")
        print(f"tflops: {perf(ms)}")

    report_latency = False
    if report_latency:
        perf = lambda ms: ms
    c = fn()
    dsize = a.nelement() * a.element_size() + b.nelement() * b.element_size() + c.nelement() * c.element_size()
    mem_bw = lambda ms: (dsize * 1e-9) / (ms * 1e-3)
    if verbose:
        print(f"{provider}:{M},{N},{K}")
        print(f"mem bw: GB/s: {mem_bw(ms)}")

    return perf(ms), perf(max_ms), perf(min_ms)

# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['dtype', 'M', 'N', 'K', 'B'],  # Argument names to use as an x-axis for the plot
#         x_vals=[
#             # (torch.bfloat16, 8192, 1280, 4096, 1),
#             # (torch.bfloat16, 1024, 8192, 4096, 1),
#             # (torch.bfloat16, 8192, 7168, 4096, 1),
#             # (torch.bfloat16, 3584, 8192, 4096, 1),
#             # (torch.bfloat16, 8192, 4096, 4096, 1),
#             # (torch.bfloat16, 8192, 4096, 1280, 1),
#             # (torch.bfloat16, 1024, 4096, 8192, 1),
#             # (torch.bfloat16, 8192, 4096, 7168, 1),
#             # (torch.bfloat16, 3584, 4096, 8192, 1),
#             # (torch.bfloat16, 8192, 4096, 4096, 1),
#             (torch.bfloat16, 4096, 4096, 128, 8),
#             # (torch.bfloat16, 4096, 4096, 8192, 1),
#             # (torch.bfloat16, 8192, 4096, 3584, 1),
#             # (torch.bfloat16, 7168, 4096, 8192, 1),
#             # (torch.bfloat16, 8192, 4096, 1024, 1),
#             # (torch.bfloat16, 1280, 4096, 8192, 1),
#             # (torch.float16, 5120, 32000, 4096, 1),
#             # (torch.float16, 13824, 5120, 4096, 1),
#             # (torch.float16, 5120, 27648, 4096, 1),
#             # (torch.float16, 5120, 5120, 4096, 1),
#             # (torch.bfloat16, 128, 4096, 4096, 8),
#             # (torch.float16, 5120, 4096, 32000, 1),
#             # (torch.float16, 13824, 4096, 5120, 1),
#             # (torch.float16, 5120, 4096, 27648, 1),
#             # (torch.float16, 5120, 4096, 5120, 1),
#             # (torch.float16, 128, 4096, 4096, 40),
#             # (torch.float16, 4096, 4096, 128, 40),
#             # (torch.bfloat16, 4096, 128, 4096, 8),
#             # (torch.bfloat16, 128, 4096, 4096, 8),
#             # (torch.float16, 128, 4096, 4096, 40),
#             # (torch.float16, 15360, 4096, 5120, 1),
#             # (torch.float16, 5120, 4096, 5120, 1),
#             # (torch.float16, 5120, 4096, 13824, 1),
#             # (torch.float16, 27648, 4096, 5120, 1),
#             # (torch.float16, 32000, 4096, 5120, 1),

#         ],  # Different possible values for `x_name`
#         line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
#         # line_vals=['matmul',  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
#         # line_names=['bmm',  'bmm-a.t-b.t', 'bmm-a.t', 'bmm-b.t'], #, "triton_bmm"],
#         line_vals=['bmm',  'bmm-a.t-b.t', 'bmm-a.t', 'bmm-b.t'], #, "triton_bmm"],
#         line_names=['bmm',  'bmm-a.t-b.t', 'bmm-a.t', 'bmm-b.t'], #, "triton_bmm"],
#         # Line styles
#         styles=[('k', '--'), ('k', '-'), ('red', '-'), ('red', '--'), ('g', '-'), ('g', '--'), ('b', '-'), ('b', '--'),
#                 ('m', '-'), ('m', '--'), ('c', '-'), ('y', '-')],
#         ylabel="TFLOPS",  # Label name for the y-axis
#         plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
#         args={},
#     ))
# def benchmark(dtype, M, N, K, B, provider, quantiles = [0.5, 0.2, 0.8]):
#     print(f"## {provider} benchmark {M}, {N}, {K}, {B} ##")
#     print(dtype)
#     if re.search('bmm', provider):
#         a = torch.randn((B, M, K), dtype=dtype, device='cuda')
#         b = torch.randn((B, K, N), dtype=dtype, device='cuda')
#     else:
#         a = torch.randn((M, K), dtype=dtype, device='cuda')
#         b = torch.randn((K, N), dtype=dtype, device='cuda')
    
#     if provider == 'bmm':
#         fn = lambda: torch.bmm(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'bmm-a.t':
#         a = torch.randn((K, M, B), device='cuda', dtype=dtype)
#         a = a.permute(2,1,0)
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'bmm-b.t':
#         b = torch.randn((N, K, B), device='cuda', dtype=dtype)
#         b = b.permute(2,1,0)
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'bmm-a.t-b.t':
#         a = torch.randn((K, M, B), device='cuda', dtype=dtype)
#         a = a.permute(2,1,0)
#         b = torch.randn((N, K, B), device='cuda', dtype=dtype)
#         b = b.permute(2,1,0)
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'matmul':
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'matmul-a.t':
#         a = torch.randn((K, M), device='cuda', dtype=dtype)
#         a = a.t()
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'matmul-b.t':
#         b = torch.randn((N, K), device='cuda', dtype=dtype)
#         b = b.t()
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'matmul-a.t-b.t':
#         a = torch.randn((K, M), device='cuda', dtype=dtype)
#         b = torch.randn((N, K), device='cuda', dtype=dtype)
#         a = a.t()
#         b = b.t()
#         fn = lambda: torch.matmul(a, b)
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
#     elif provider == 'triton' or provider == 'triton_matmul':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b), quantiles=quantiles)
#         fn = lambda: triton_matmul(a, b)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     # return perf(ms), perf(max_ms), perf(min_ms)
#     if verbose:
#         print(f"latency: ms : {ms}")
#         print(f"tflops: {perf(ms)}")

#     report_latency = True
#     if report_latency:
#         perf = lambda ms: ms
#     c = fn()
#     dsize = a.nelement() * a.element_size() + b.nelement() * b.element_size() + c.nelement() * c.element_size()
#     mem_bw = lambda ms: (dsize * 1e-9) / (ms * 1e-3)
#     if verbose:
#         print(f"{provider}:{M},{N},{K}")
#         print(f"mem bw: GB/s: {mem_bw(ms)}")

#     return perf(ms), perf(max_ms), perf(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K', 'B'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            ( 4096, 4096, 128, 8),
            ( 128, 4096, 4096, 8),
            ( 128, 4096, 4096, 40),
            ( 4096, 4096, 128, 40),
            ( 4096, 128, 4096, 8),
            ( 128, 4096, 4096, 8),
            ( 128, 4096, 4096, 40),
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['bmm'],
        line_names=['bmm'],
        # line_vals=['matmul-a.t'],#,  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        # line_names=['matmul-a.t'],#,  'matmul-a.t-b.t', 'matmul-a.t', 'matmul-b.t'], #, "triton_matmul"],
        # Line styles
        styles=[('k', '--'), ('k', '-'), ('red', '-'), ('red', '--'), ('g', '-'), ('g', '--'), ('b', '-'), ('b', '--'),
                ('m', '-'), ('m', '--'), ('c', '-'), ('y', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="bmm-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def bbenchmark(M, N, K, B, provider, dtype=torch.bfloat16, quantiles = [0.5, 0.2, 0.8]):
    print(f"## {provider} benchmark {M}, {N}, {K} ##")
    print(dtype)
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    b = torch.randn((K, N), dtype=dtype, device='cuda')
    # b = torch.randn((N, K), device='cuda', dtype=dtype)
    # b = b.t()
    
    if provider == 'bmm':
        a = torch.randn((B, M, K), dtype=dtype, device='cuda')
        b = torch.randn((B, K, N), dtype=dtype, device='cuda')
        fn = lambda: torch.bmm(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul':
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-a.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        a = a.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-b.t':
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'matmul-a.t-b.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        a = a.t()
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'triton' or provider == 'triton_matmul':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b), quantiles=quantiles)
        fn = lambda: triton_matmul(a, b)
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    if verbose:
        print(f"latency: ms : {ms}")
        print(f"tflops: {perf(ms)}")

    report_latency = False
    if report_latency:
        perf = lambda ms: ms
    c = fn()
    dsize = a.nelement() * a.element_size() + b.nelement() * b.element_size() + c.nelement() * c.element_size()
    mem_bw = lambda ms: (dsize * 1e-9) / (ms * 1e-3)
    if verbose:
        print(f"{provider}:{M},{N},{K}")
        print(f"mem bw: GB/s: {mem_bw(ms)}")

    return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run()
benchmark.run(show_plots=True, print_data=True, save_path=".")
# bbenchmark.run(show_plots=True, print_data=True, save_path=".")
