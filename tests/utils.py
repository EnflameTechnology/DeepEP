import argparse
import inspect
import json
import tempfile
from pathlib import Path
from this import d
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免DISPLAY环境变量错误
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import torch
import torch.distributed as dist
import random
import statistics
from typing import Optional, Union, List, Dict, Tuple, Callable
import torch_gcu
from torch_gcu import transfer_to_gcu

# #########################################################################################
# #############  Shared argparse helpers — eliminate duplicated parser boilerplate  ######
# #########################################################################################
# Each `add_*` helper binds the `help` text to the actual `default` via f-strings, so
# the "(default: X)" text can never drift out of sync with the real default value again
# (a class of bugs that previously infected ~half the tests in this directory).

def build_base_parser(description: str, *, defaults: dict = None,
                      num_topk_groups: bool = False,
                      num_topk: bool = True) -> argparse.ArgumentParser:
    """Create a parser preloaded with the common EP test parameters.

    `defaults` may override any of: num_processes, num_tokens, hidden, num_topk,
    num_experts. Pass `num_topk_groups=True` to also add `--num-topk-groups`.
    Pass `num_topk=False` to omit `--num-topk` (for scripts that use their own
    top-k flag, e.g. `--topk`).
    """
    d = {'num_processes': 8, 'num_tokens': 128, 'hidden': 7168,
         'num_topk': 8, 'num_experts': 256}
    if defaults:
        d.update(defaults)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--num-processes', type=int, default=d['num_processes'],
                        help=f'Number of processes to spawn (default: {d["num_processes"]})')
    parser.add_argument('--num-tokens', type=int, default=d['num_tokens'],
                        help=f'Number of tokens (default: {d["num_tokens"]})')
    parser.add_argument('--hidden', type=int, default=d['hidden'],
                        help=f'Hidden dimension size (default: {d["hidden"]})')
    if num_topk_groups:
        parser.add_argument('--num-topk-groups', type=int, default=None,
                            help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    if num_topk:
        parser.add_argument('--num-topk', type=int, default=d['num_topk'],
                            help=f'Number of top-k experts (default: {d["num_topk"]})')
    parser.add_argument('--num-experts', type=int, default=d['num_experts'],
                        help=f'Number of experts (default: {d["num_experts"]})')
    return parser


def add_sms_arg(parser: argparse.ArgumentParser, default: int = 24) -> None:
    parser.add_argument('--num-sms', type=int, default=default,
                        help=f'Number of SMs (default: {default})')


def add_hardcode_local_ranks_arg(parser: argparse.ArgumentParser, default=None) -> None:
    parser.add_argument('--num-hardcode-local-ranks', type=int, default=default,
                        help='Set EP_INTRA_RANKS for single-node internode simulation '
                             '(e.g., 8 processes + 4 local ranks => 2 simulated nodes). '
                             'Default: auto when WORLD_SIZE=1')


def add_low_latency_flags(parser: argparse.ArgumentParser, *, test_fp8: bool = False,
                          mnnvl: bool = True) -> None:
    """MNNVL/NVLink/LogFMT flags shared by the low-latency tests."""
    if mnnvl:
        parser.add_argument('--allow-mnnvl', action='store_true',
                            help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    if test_fp8:
        parser.add_argument('--test-fp8', action='store_true',
                            help='Whether to test FP8 configurations')


def add_iteration_seed_args(parser: argparse.ArgumentParser) -> None:
    """CLI args for the per-seed iteration count and the seed count.

    Both default to ``None`` so each caller keeps its own fallback semantics:
    the hash-consistency loops use ``NUM_ITERATIONS`` iterations per seed and
    loop over ``NUM_SEEDS`` seeds. A large ``--num-seeds`` value (e.g.
    1000000000) effectively acts as the former pressure test.
    """
    parser.add_argument('--num-iterations', type=int, default=None,
                        help='Number of hash-consistency iterations per seed '
                             '(default: 50 in test_low_latency.py and '
                             'test_intranode.py; 300 in '
                             'test_low_latency_dispatch.py and '
                             'test_low_latency_combine.py)')
    parser.add_argument('--num-seeds', type=int, default=None,
                        help='Number of seeds to test; a large value (e.g. '
                             '1000000000) effectively runs a pressure test '
                             '(default: 2 in test_low_latency.py and '
                             'test_intranode.py; 3 in '
                             'test_low_latency_dispatch.py and '
                             'test_low_latency_combine.py)')


def add_imbalance_arg(parser: argparse.ArgumentParser, *,
                      imbalance_default: int = 0) -> None:
    """CLI args for the three imbalance dimensions of the dispatch tests.

    1. Token-level imbalance (--token-imbalance-ratio): non-uniform number of
       tokens across ranks (in percent). 0 = default moderate skew, 200 = heavy
       skew; drives the non-uniform token test.
    2. Expert-level imbalance (--expert-imbalance-ratio): a fraction (0.0-1.0)
       of the top-k slots that "hot" experts occupy. Hot experts (num_hot =
       ratio * num_topk) absorb almost all tokens via a fixed large score bias,
       and an equal number of cold experts receive 0 tokens. 0.0 = disabled.
    3. Time-level imbalance (--host-jitter / --device-jitter): random per-rank
       delays of 0-10 ms to emulate stragglers.
    """
    parser.add_argument('--token-imbalance-ratio', type=int, default=imbalance_default,
                        help='Token-level imbalance ratio (in percent) for the non-uniform '
                             'per-rank token distribution test '
                             '(0 = default moderate skew, 200 = heavy skew). Default: 0')
    parser.add_argument('--expert-imbalance-ratio', type=float, default=0.0,
                        help='Expert-level imbalance ratio (0.0-1.0): num_hot = ratio * num_topk '
                             'hot experts receive almost all tokens and an equal number of cold '
                             'experts get 0 tokens. 0.0 = disabled (default: 0.0)')
    parser.add_argument('--host-jitter', action='store_true',
                        help='Time-level imbalance: enable host-side jitter (random 0-10 ms '
                             'per rank). Uses std::this_thread::sleep_for (blocks CPU thread). '
                             'Can be combined with --device-jitter.')
    parser.add_argument('--device-jitter', action='store_true',
                        help='Time-level imbalance: enable device-side jitter (random 0-10 ms '
                             'per rank). Uses a GPU delay kernel via tops::__nanosleep '
                             '(does not block CPU). Can be combined with --host-jitter.')


def add_fp8_dispatch_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dispatch-use-fp8', action='store_true',
                        help='Use FP8 format for dispatch (default: False, uses BF16)')


def add_ib_args(parser: argparse.ArgumentParser, *, imbalance_default: int = 0,
                total_tokens_help: str = None, add_run_mode: bool = False) -> None:
    """Imbalance-test args: --total-tokens, --imbalance-ratio, optionally --run-mode."""
    help_total = total_tokens_help or (
        'Total tokens across all ranks. Defaults to num-tokens * num-ranks if not specified.')
    parser.add_argument('--total-tokens', type=int, default=None, help=help_total)
    parser.add_argument('--imbalance-ratio', type=int, default=imbalance_default,
                        help=f'Imbalance ratio in percent (default: {imbalance_default})')
    if add_run_mode:
        parser.add_argument('--run-mode', choices=['single', 'list'], default='single',
                            help='single (default): one run using --total-tokens and '
                                 '--imbalance-ratio; list: iterate over the cartesian product '
                                 'of SWEEP_TOTAL_TOKENS_LIST × SWEEP_IMBALANCE_RATIO_LIST '
                                 'defined at the top of this script.')


def add_kineto_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--use-kineto', action='store_true',
                        help='Use bench_kineto() instead of bench() for performance measurement (default: False)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for performance JSON data. If not specified, no JSON file is saved.')


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'eccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'gcu:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('gcu')
    torch.gcu.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

# UserWarning: GCU not support Double use Float replace
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    # 处理极大值：如果值太大会导致平方溢出，先归一化
    x_max = x.abs().max().item()
    y_max = y.abs().max().item()
    scale = max(x_max, y_max)
    
    # 如果最大值超过 1e19，归一化避免平方溢出
    # (1e19)^2 = 1e38，接近 float32 的极限
    if scale > 1e19:
        x_norm = (x / scale).float() + 1e-10
        y_norm = (y / scale).float() + 1e-10
    else:
        # GCU 不支持 double，直接用 float
        x_norm = x.float() + 1
        y_norm = y.float() + 1
    
    denominator = (x_norm * x_norm + y_norm * y_norm).sum()
    sim = 2 * (x_norm * y_norm).sum() / denominator
    return (1 - sim).item()


def align_up(x, y):
    return (x + y - 1) // y * y


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align_up(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, aligned_n//128, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, aligned_n//128).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, aligned_n//128)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align_up(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:,:n].contiguous()


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.gcu.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='gcu')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.gcu.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.gcu.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.gcu.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


def bench_separate(phase_fns: Dict[str, Callable], num_warmups: int = 10, num_tests: int = 10,
                   num_amortize: int = 30,
                   phase_setup_fns: Optional[Dict[str, Callable]] = None,
                   phase_post_fns: Optional[Dict[str, Callable]] = None):
    """Benchmark multiple phases with two-level timing for accuracy and statistics.

    Two-level loop design:
      for round in range(num_tests):          # outer: num_tests independent samples
          start.record()
          for _ in range(num_amortize):       # inner: amortize ~7us/event overhead
              [setup_fn()]                    # optional, amortized alongside fn
              phase_fn()
              [post_fn()]                     # optional, e.g. async wait
          end.record()
          round_time = elapsed / num_amortize

    Each rank stores its own local latency per round. Cross-rank aggregation
    (collective completion time = slowest rank) is performed by the caller via
    all_gather + aggregate_bench_separate_results, whose summary_max_per_round
    field reports max(all_ranks) per round — identical semantics to
    analyse_kernel_durations.

    Event overhead per sample ≈ 14 us / num_amortize (≈ 0.28 us at 50).
    num_tests rounds enable avg/min/max statistics and confidence intervals.

    Args:
        phase_fns: Ordered dict mapping phase name to callable to be timed.
        num_warmups: Warmup iterations before measurement (per phase).
        num_tests: Number of independent measurement rounds (outer loop).
        num_amortize: Iterations per round for event-overhead amortization
                      (inner loop).
        phase_setup_fns: Optional dict of callables run before each inner
                         iteration. Cost is amortized alongside fn; keep cheap
                         relative to fn to avoid measurement bias.
        phase_post_fns: Optional dict of callables run after each inner
                        iteration, e.g. to drain async operations.

    Returns:
        Dict mapping phase name to numpy array of per-round local durations
        in seconds. Length = num_tests (one value per measurement round).
        Use aggregate_bench_separate_results(…).summary_max_per_round for
        the collective completion time (max across all ranks per round).
    """
    torch.gcu.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='gcu')

    phase_setup_fns = phase_setup_fns or {}
    phase_post_fns = phase_post_fns or {}
    phase_names = list(phase_fns.keys())
    result = {}

    for name in phase_names:
        setup_fn = phase_setup_fns.get(name)
        fn = phase_fns[name]
        post_fn = phase_post_fns.get(name)

        for _ in range(num_warmups):
            if setup_fn is not None:
                setup_fn()
            fn()
            if post_fn is not None:
                post_fn()

        cache.zero_()

        round_times = []
        for _ in range(num_tests):
            start = torch.gcu.Event(enable_timing=True)
            end = torch.gcu.Event(enable_timing=True)
            start.record()
            for _ in range(num_amortize):
                if setup_fn is not None:
                    setup_fn()
                fn()
                if post_fn is not None:
                    post_fn()
            end.record()
            torch.gcu.synchronize()
            round_times.append(start.elapsed_time(end) / 1e3 / max(num_amortize, 1))

        result[name] = np.array(round_times, dtype=np.float64)

    return result


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names: Union[str, tuple], num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1, return_stats: bool = False):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for i in range(3):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='gcu')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='gcu')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='gcu'))
                for _ in range(num_tests):
                    fn()
                torch.gcu.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)
        profile_data = json.loads(Path(trace_path).read_text())
    # Extract all kernel durations from trace data for accurate min/max calculation
    else:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

    # Process each kernel
    kernel_results = []
    for kernel_name in kernel_names:
        events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]
        events = sorted(events, key=lambda event: event['ts'])
        durations = [event['dur'] / 1e6 for event in events]  # Convert from microseconds to seconds

        if num_kernels_per_period > 1:
            # Group durations by period
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            period_durations = []
            for j in range(num_kernels_per_period):
                # Extract durations for this period component (e.g., all send times or all recv times)
                component_durations = [durations[k] for k in range(j, len(durations), num_kernels_per_period)]
                period_durations.append(component_durations)

            if return_stats:
                # Return stats for each period component
                period_stats = []
                for component_dur_list in period_durations:
                    period_stats.append({
                        'avg': np.average(component_dur_list),
                        'min': np.min(component_dur_list),
                        'max': np.max(component_dur_list)
                    })
                kernel_results.append(period_stats)
            else:
                # Return average for each period component (backward compatibility)
                # Match original behavior: sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                kernel_results.append([np.average(component_dur_list) for component_dur_list in period_durations])
        else:
            # Single kernel per period
            if return_stats:
                kernel_results.append({
                    'avg': np.average(durations),
                    'min': np.min(durations),
                    'max': np.max(durations)
                })
            else:
                # Return average (backward compatibility)
                kernel_results.append(np.average(durations))

    # Return execution durations
    if return_stats:
        return kernel_results if is_tuple else kernel_results[0]
    else:
        return kernel_results if is_tuple else kernel_results[0]


def draw_bandwidth_chart(bandwidth_data: List[Dict], output_path: Optional[str] = None, title: str = "Bandwidth Comparison"):
    from collections import defaultdict

    if not bandwidth_data:
        raise ValueError("bandwidth_data is empty")

    # 按 hidden size 分组数据，并按 num_tokens 排序
    data_by_hidden = defaultdict(lambda: {'tokens': [], 'bandwidth': []})

    for item in bandwidth_data:
        hidden = item['hidden']
        num_tokens = item['num_tokens']
        bandwidth = item['bandwidth']

        data_by_hidden[hidden]['tokens'].append(num_tokens)
        data_by_hidden[hidden]['bandwidth'].append(bandwidth)

    # 对每个 hidden size 的数据按 num_tokens 排序
    hidden_sizes = sorted(data_by_hidden.keys())
    for hidden in hidden_sizes:
        data = data_by_hidden[hidden]
        # 按 tokens 排序
        sorted_indices = sorted(range(len(data['tokens'])), key=lambda i: data['tokens'][i])
        data['tokens'] = [data['tokens'][i] for i in sorted_indices]
        data['bandwidth'] = [data['bandwidth'][i] for i in sorted_indices]

    # 使用colormap生成足够的颜色
    num_lines = len(hidden_sizes)
    cmap = plt.cm.get_cmap('tab10')
    if num_lines > 0:
        colors = [cmap(i / max(num_lines - 1, 1)) for i in range(num_lines)]
    else:
        colors = []

    # 动态生成线条数据
    lines_data = []
    for idx, hidden_size in enumerate(hidden_sizes):
        data = data_by_hidden[hidden_size]
        lines_data.append({
            'name': f'hidden_{hidden_size}',
            'tokens': data['tokens'],
            'bandwidth': data['bandwidth'],
            'color': colors[idx],
            'marker': 'o',
            'linestyle': '-'
        })

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制每条线
    for line in lines_data:
        tokens = line['tokens']
        bandwidth_values = line['bandwidth']

        # 绘制带宽线
        marker = line.get('marker', 'o')
        linestyle = line.get('linestyle', '-')
        plt.plot(tokens, bandwidth_values, color=line['color'], marker=marker,
                linestyle=linestyle, markersize=5, label=f"{line['name']}")

        # 在每个数据点上显示纵轴数值
        for x, y in zip(tokens, bandwidth_values):
            # 格式化数值，保留2位小数
            text = f'{y:.2f}'
            # 将文本放在点的上方，稍微偏移以避免遮挡
            plt.text(x, y, text, ha='center', va='bottom', fontsize=8, color=line['color'])

    # 设置图表属性
    plt.xlabel('Number of Tokens')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    #plt.xscale('log', basex=2)  # 使用对数刻度以便更好地显示token数量的增长

    # 调整布局
    plt.tight_layout()

    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    else:
        plt.show()

    # 关闭图表以释放内存
    plt.close()


def draw_latency_chart(latency_data: List[Dict], output_path: Optional[str] = None, title: str = "Latency Comparison", use_log_scale: bool = True):
    from collections import defaultdict

    if not latency_data:
        raise ValueError("latency_data is empty")

    # 按 hidden size 分组数据，并按 num_tokens 排序
    data_by_hidden = defaultdict(lambda: {'tokens': [], 'avg': [], 'min': [], 'max': []})

    for item in latency_data:
        hidden = item['hidden']
        num_tokens = item['num_tokens']

        data_by_hidden[hidden]['tokens'].append(num_tokens)
        data_by_hidden[hidden]['avg'].append(item['avg'])
        data_by_hidden[hidden]['min'].append(item['min'])
        data_by_hidden[hidden]['max'].append(item['max'])

    # 对每个 hidden size 的数据按 num_tokens 排序
    hidden_sizes = sorted(data_by_hidden.keys())
    for hidden in hidden_sizes:
        data = data_by_hidden[hidden]
        # 按 tokens 排序
        sorted_indices = sorted(range(len(data['tokens'])), key=lambda i: data['tokens'][i])
        data['tokens'] = [data['tokens'][i] for i in sorted_indices]
        data['avg'] = [data['avg'][i] for i in sorted_indices]
        data['min'] = [data['min'][i] for i in sorted_indices]
        data['max'] = [data['max'][i] for i in sorted_indices]

    # 使用colormap生成足够的颜色
    num_lines = len(hidden_sizes)
    cmap = plt.cm.get_cmap('tab10')
    if num_lines > 0:
        colors = [cmap(i / max(num_lines - 1, 1)) for i in range(num_lines)]
    else:
        colors = []

    # 动态生成线条数据
    lines_data = []
    for idx, hidden_size in enumerate(hidden_sizes):
        data = data_by_hidden[hidden_size]
        lines_data.append({
            'name': f'hidden_{hidden_size}',
            'tokens': data['tokens'],
            'avg': data['avg'],
            'min': data['min'],
            'max': data['max'],
            'color': colors[idx],
            'marker': 'o',
            'linestyle': '-'
        })

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制每条带宽图
    for line in lines_data:
        tokens = line['tokens']
        avg_values = line['avg']
        # 绘制填充区域（min到max）
        plt.fill_between(tokens, line['min'], line['max'], color=line['color'], alpha=0.2, label=f"Min Max")
        # 绘制中间线（avg），但不显示在图例中
        marker = line.get('marker', 'o')
        linestyle = line.get('linestyle', '-')
        plt.plot(tokens, avg_values, color=line['color'], marker=marker, linestyle=linestyle, markersize=5, label=f"{line['name']}")

        # 在每个数据点上显示纵轴数值
        for x, y in zip(tokens, avg_values):
            # 格式化数值，保留1位小数
            text = f'{y:.1f}'
            # 将文本放在点的上方，稍微偏移以避免遮挡
            plt.text(x, y, text, ha='center', va='bottom', fontsize=8, color=line['color'])

    # 设置图表属性
    plt.xlabel('Number of Tokens')
    plt.ylabel('Execution Time (us)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    if use_log_scale:
        plt.xscale('log', base=2)  # 使用对数刻度以便更好地显示token数量的增长

    # 调整布局
    plt.tight_layout()

    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    else:
        plt.show()

    # 关闭图表以释放内存
    plt.close()

def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()


def collect_dispatch_combine_results(
    packed_recv_x, packed_recv_count, handle,
    combined_x: torch.Tensor,
    num_local_experts: int, num_ranks: int,
    dispatch_use_fp8: bool, to_cpu: bool = True,
) -> dict:
    """Collect dispatch + combine raw tensors into a dict.

    Packs low_latency_dispatch and low_latency_combine outputs into a
    single dict (optionally transferred to CPU).  Derived metrics like
    diff or diagnostics are left to the caller.

    Args:
        packed_recv_x: dispatch output; tuple (fp8_data, scales) if FP8,
            single tensor if BF16.
        packed_recv_count: per-expert received token count list.
        handle: dispatch routing handle (recv_src_info, recv_layout_list).
        combined_x: low_latency_combine output tensor.
        num_local_experts: number of local experts on this rank.
        num_ranks: total number of ranks.
        dispatch_use_fp8: whether dispatch used FP8 quantization.
        to_cpu: True (default) to transfer to CPU; False to keep on device.

    Returns:
        dict with keys: dispatch_recv_x_fp8/scales or dispatch_recv_x,
        recv_counts (E, R), recv_begins (E, R), recv_count (E,),
        recv_src_info, combine_x.
    """
    _t = (lambda t: t.cpu()) if to_cpu else (lambda t: t)
    result = {}

    # --- dispatch data ---
    if dispatch_use_fp8:
        result['dispatch_recv_x_fp8'] = _t(packed_recv_x[0])
        result['dispatch_recv_x_scales'] = _t(packed_recv_x[1])
    else:
        result['dispatch_recv_x'] = _t(packed_recv_x)

    # --- routing metadata (GCU layout: view as (num_ranks, 2)) ---
    per_expert_counts, per_expert_begins, per_expert_recv_count = [], [], []
    for ei in range(num_local_experts):
        rlr = handle[1][ei].view(num_ranks, 2)
        per_expert_counts.append(_t(rlr[:, 0]))
        per_expert_begins.append(_t(rlr[:, 1]))
        per_expert_recv_count.append(_t(packed_recv_count[ei]))
    result['recv_counts'] = torch.stack(per_expert_counts)
    result['recv_begins'] = torch.stack(per_expert_begins)
    result['recv_count'] = torch.stack(per_expert_recv_count)
    result['recv_src_info'] = _t(handle[0])

    # --- combine ---
    result['combine_x'] = _t(combined_x)

    return result


def get_imbalance_token_counts(ranks: int, total_tokens: int, imbalance_ratio: float, seed: int) -> list[int]:
    """
    Generate a list of token counts for each rank with a specified imbalance ratio.
    :param ranks: Number of ranks
    :param total_tokens: Total number of tokens to distribute
    :param imbalance_ratio: Ratio of imbalance
    :return: List of token counts for each rank
    """
    random.seed(seed)
    avg_count = (total_tokens + ranks - 1) // ranks # avoid total_tokens < ranks thus base_count = 0

    if imbalance_ratio == 0:
        counts = [total_tokens // ranks] * ranks
        for i in range(total_tokens % ranks):
            counts[i] += 1
        random.shuffle(counts)
        return counts

    max_random_value = int(math.ceil(avg_count * (1.0 + imbalance_ratio)))
    total_assigned = 0
    counts = [0] * ranks
    for i in range(ranks):
        rand_num = random.randint(0, max_random_value)
        # rand_num = max(0, rand_num - 1)  # since we already assigned 1 token to each rank
        if total_assigned + rand_num >= total_tokens:
            rand_num = total_tokens - total_assigned
            counts[i] += rand_num
            total_assigned += rand_num
            break
        else:
            counts[i] += rand_num
            total_assigned += rand_num
    # if generated count did not match total tokens, adjust last rank's count to fill the gap
    if total_assigned < total_tokens:
        add = (total_tokens - total_assigned) // ranks
        remainder = total_tokens - total_assigned - add*ranks
        for i in range(ranks):
            counts[i] += add
            if remainder > 0:
                counts[i] += 1
                remainder -= 1
    # shuffle for random sequence
    random.shuffle(counts)
    return counts

def analyse_kernel_durations(trace_name_template: str, ranks: 'int | list[int]', kernel_names: list[str],
                             verbose: bool = True):
    # trace_name_template example: "kineto_trace_rank{rank}_20251225T054459348.json"
    # read traces from all ranks, and log each kernel call durations, which is max time of all ranks
    # return a list of all kernel durations (there can be multiple calls of the same kernel)
    rank_list = list(range(ranks)) if isinstance(ranks, int) else ranks

    trace_list = []
    for rank_id in rank_list:
        trace_path = trace_name_template.format(rank=rank_id)
        with open(trace_path, 'r') as f:
            if verbose:
                print(f"Analyzing trace for rank {rank_id} from file: {trace_path}")
            trace_list.append(json.load(f))

    # Process each kernel
    kernel_results = {}
    for kernel_name in kernel_names:
        # read the events from ranks first. For each rank data, sort the data by timestamp,
        # so that we can align the same kernel instances across ranks.
        events_of_ranks = []
        for idx, rank_id in enumerate(rank_list):
            events = [event for event in trace_list[idx]['traceEvents'] if f'::{kernel_name}' in event['name']]
            events = sorted(events, key=lambda event: event['ts'])
            events_of_ranks.append(events)

        all_durations = []
        for i in range(len(events_of_ranks[0])):
            # max duration method
            durations_in_ranks = []
            for idx in range(len(rank_list)):
                event = events_of_ranks[idx][i]
                duration = event['dur']*1000  # dur in microseconds
                durations_in_ranks.append(duration)

            if verbose:
                print(f"Kernel {kernel_name}, Instance {i}, durations: {durations_in_ranks} ns")
                print(f"Kernel {kernel_name}, Instance {i}, max duration: {max(durations_in_ranks)} ns")
            all_durations.append(max(durations_in_ranks))

        durations_avg = statistics.mean(all_durations)
        durations_min = min(all_durations)
        durations_max = max(all_durations)
        kernel_results[kernel_name] = {
            'latency_us': durations_avg / 1000,
            'latency_us_min': durations_min / 1000,
            'latency_us_max': durations_max / 1000,
        }

    return kernel_results


def aggregate_bench_separate_results(result_path_template: str, ranks: 'int | list[int]',
                                     kernel_names: Optional[List[str]] = None):
    """Aggregate per-rank bench_separate JSON results across ranks.

    Each per-rank JSON is expected to contain keys like ``{kernel_name}_times_us``
    (list of floats, one per measurement round — output of bench_separate).

    Returns a dict keyed by kernel name, each containing:
      - per_round: list of dicts with per-round cross-rank stats
      - summary_max_per_round: collective completion time
          For each round take max latency across all ranks (= the slowest rank,
          which determines when the collective operation actually finishes), then
          compute avg/min/max across rounds. This is the primary performance
          metric and matches the semantics of analyse_kernel_durations.
      - summary_all: flatten all (rank * round) values, then avg/min/max.
          Reflects average-rank latency distribution; lower than
          summary_max_per_round since fast ranks pull the mean down.
    """
    if kernel_names is None:
        kernel_names = ['dispatch', 'combine']
    rank_list = list(range(ranks)) if isinstance(ranks, int) else ranks

    all_rank_data = {}
    for rank_id in rank_list:
        path = result_path_template.format(rank=rank_id)
        with open(path, 'r') as f:
            print(f"Reading bench_separate results for rank {rank_id} from: {path}")
            all_rank_data[rank_id] = json.load(f)

    results = {}
    for kernel_name in kernel_names:
        key = f'{kernel_name}_times_us'
        rank_times = {r: all_rank_data[r][key] for r in rank_list}
        num_rounds = len(rank_times[rank_list[0]])

        per_round = []
        max_per_round = []
        all_flat = []
        for i in range(num_rounds):
            round_vals = [rank_times[r][i] for r in rank_list]
            per_round.append({
                'round': i,
                'rank_durations_us': round_vals,
                'avg_us': statistics.mean(round_vals),
                'min_us': min(round_vals),
                'max_us': max(round_vals),
            })
            max_per_round.append(max(round_vals))
            all_flat.extend(round_vals)

        results[kernel_name] = {
            'per_round': per_round,
            'summary_max_per_round': {
                'description': 'For each round take max across ranks, then stats across rounds',
                'avg_us': statistics.mean(max_per_round),
                'min_us': min(max_per_round),
                'max_us': max(max_per_round),
            },
            'summary_all': {
                'description': 'Flatten all rank*round durations, then overall stats',
                'avg_us': statistics.mean(all_flat),
                'min_us': min(all_flat),
                'max_us': max(all_flat),
            },
        }

    return results


def get_performance_data(file_path: str, test_name: str, data: dict) -> dict:
    """
    Parse performance data and append to JSON file
    
    Args:
        file_path: Output directory path
        test_name: Test name as key for storing the data
        data: Dictionary containing performance data in the following format:
            {
                'mode': 'low_latency',
                'use_fp8': False,
                'num_tokens': 128,
                'hidden': 7168,
                'num_topk': 8,
                'num_experts': 288,
                'num_ranks': 8,
                'dispatch_latency_us_mean': 33.46,
                'dispatch_bandwidth_mean': 9.99,
                'combine_latency_us_mean': 27.50,
                'combine_bandwidth_mean': 12.26,
                'dispatch': {...},
                'combine': {...}
            }
    
    Returns:
        The complete data dictionary including old and new data
    """
    # Build output filename
    mode = data[test_name].get('mode', 'unknown')
    output_file = os.path.join(file_path, f'{mode}_performance_data.json')

    # Ensure output directory exists
    if os.path.exists(file_path):
        print(f'Output directory already exists: {file_path}')
    else:
        os.makedirs(file_path)
        print(f'Created output directory: {file_path}')
    
    # Read existing data from file if it exists
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f'Loaded existing data from: {output_file}')
        except json.JSONDecodeError:
            print(f'Warning: Failed to parse existing file, creating new one')
            existing_data = {}
    else:
        print(f'Created new file: {output_file}')
    
    # Add timestamp to new data
    from datetime import datetime
    data[test_name]['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Append new data: if key already exists, convert value to list and append
    if test_name in existing_data:
        old_value = existing_data[test_name]
        if isinstance(old_value, list):
            old_value.append(data[test_name])
        else:
            existing_data[test_name] = [old_value, data[test_name]]
        print(f'Key "{test_name}" already exists, appended as list (now {len(existing_data[test_name])} entries)')
    else:
        existing_data[test_name] = data[test_name]
    
    # Save updated data to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f'Performance data appended to: {output_file} (key: {test_name})')
    
    return existing_data


def create_test_data(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                     rank: int, num_ranks: int, use_logfmt: bool = False, seed: int = 0,
                     expert_imbalance_ratio: float = 0.0):
    """Create representative test tensors for dispatch/combine performance or correctness tests.

    The topk_idx is built by the shared interface (``build_topk_idx``), so the
    expert-level imbalance (hot experts absorbing almost all tokens) is
    supported via ``expert_imbalance_ratio``.

    Args:
        num_tokens: Actual number of tokens for this rank (may be 0).
        hidden: Hidden dimension.
        num_experts: Total number of experts across all ranks.
        num_topk: Top-k value for expert selection.
        rank: Current rank index.
        num_ranks: Total number of ranks.
        use_logfmt: Unused; kept for API compatibility.
        seed: Random seed.
        expert_imbalance_ratio: Hot/cold expert fraction of the top-k slots
            (0.0-1.0). 0.0 disables the hot/cold construction.

    Returns:
        Tuple of (x, topk_idx, topk_weights), all on 'gcu' device.
        When ``num_tokens == 0``, topk tensors have shape ``(0, num_topk)``.
    """
    import torch
    import random
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0, 'num_experts must be divisible by num_ranks'

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1

    if num_tokens > 0:
        topk_idx = build_topk_idx(num_tokens, num_experts, num_topk,
                                  expert_imbalance_ratio=expert_imbalance_ratio)
        topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()
    else:
        topk_idx = torch.empty((0, num_topk), dtype=torch.long, device='gcu')
        topk_weights = torch.empty((0, num_topk), dtype=torch.float32, device='gcu')

    return x, topk_idx, topk_weights


def build_topk_idx(num_tokens: int, num_experts: int, num_topk: int, *,
                   device: str = 'gcu', expert_imbalance_ratio: float = 0.0,
                   num_hot: Optional[int] = None, sorted_: bool = True) -> torch.Tensor:
    """Build per-token top-k expert indices.

    Uniform (default): scores ~ |N(0,1)| + 1, then torch.topk per row.
    Expert imbalance: ``num_hot = int(ratio * num_topk)`` (or the explicit
    ``num_hot`` argument) hot experts are sampled first and get a large score
    bias (10x the mean of the score matrix, far above the max non-hot score),
    so every token selects all of them and they absorb nearly all tokens.
    Then ``num_cold = int(ratio * num_topk)`` cold experts are picked from the
    remaining pool (never overlapping the hot ones) and their scores are
    pinned to 0, so they are never selected. The leftover experts compete
    for the remaining ``num_topk - num_hot`` slots per token.

    Args:
        num_tokens: Number of tokens (may be 0).
        num_experts: Total number of experts.
        num_topk: Top-k value for expert selection.
        device: Target device for the returned tensor.
        expert_imbalance_ratio: Hot/cold expert fraction of the top-k slots
            (0.0-1.0). 0.0 disables the hot/cold construction.
        num_hot: Number of hot experts (default: int(ratio * num_topk)).
        sorted_: Whether top-k results are returned in descending score order.

    Returns:
        topk_idx of shape (num_tokens, num_topk), dtype int64, on ``device``.
        When ``num_tokens == 0``, returns an empty (0, num_topk) tensor.
    """
    if num_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.long, device=device)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device).abs() + 1

    if expert_imbalance_ratio > 0:
        num_hot = min(max(int(int(expert_imbalance_ratio * num_topk) if num_hot is None else num_hot), 1), num_experts)
        hot_indices = random.sample(range(num_experts), num_hot)
        hot_bias = 10.0 * scores.mean().item()  # 10x the mean of the score matrix; must exceed max non-hot score (~6); .item() -> Python scalar for GCU scatter_
        bias = torch.zeros(num_experts, dtype=torch.float32, device=device).scatter_(
            0, torch.tensor(hot_indices, dtype=torch.long, device=device), hot_bias)
        scores = scores + bias.unsqueeze(0)

        # Cold experts: pin scores to 0, never enter the top-k. Sampled from the
        # pool excluding the already-picked hot indices.
        num_cold = min(int(expert_imbalance_ratio * num_topk), num_experts - num_hot)
        cold_indices = []
        if num_cold > 0:
            pool = [e for e in range(num_experts) if e not in set(hot_indices)]
            cold_indices = random.sample(pool, num_cold)
            assert set(cold_indices).isdisjoint(hot_indices), \
                f'cold {cold_indices} overlaps hot {hot_indices}'
            cold_idx = torch.tensor(cold_indices, dtype=torch.long, device=device)
            scores.index_fill_(1, cold_idx, 0.0)  # avoid advanced-indexing assignment, risky on GCU
        print(f'[build_topk_idx] tokens={num_tokens} experts={num_experts} topk={num_topk} '
              f'hot={num_hot} {hot_indices} cold={num_cold} {cold_indices}')
    return torch.topk(scores, num_topk, dim=-1, largest=True, sorted=sorted_)[1]
