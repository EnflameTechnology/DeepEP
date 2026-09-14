#!/usr/bin/env python3
"""
Run topsprof profiling and parse the resulting CSV trace file.
Supports multi-node: parses all node CSVs and merges results.

Usage examples:
    # Single node: auto-generate csv/json filenames from --python-args
    # csv → node0_ranks8_tokens64_hidden7168_experts256_topk8_ibratio0.csv
    # json → ranks8_tokens64_hidden7168_experts256_topk8_ibratio0.json
    python3 run_topsprof.py \
        --python-script test_low_latency_benchmark_topsprof.py \
        --python-args "--num-processes 8 --num-tokens 8 --hidden 7168 --num-topk 8 --num-experts 256 --dispatch-use-fp8"

    # Multi-node: each node generates its own csv, final json merges all nodes
    # node0 csv → node0_ranks16_tokens64_hidden7168_experts256_topk8_ibratio40.csv
    # node1 csv → node1_ranks16_tokens64_hidden7168_experts256_topk8_ibratio40.csv
    # json → ranks16_tokens64_hidden7168_experts256_topk8_ibratio40.json
    python3 run_topsprof.py \
        --python-script test_low_latency_benchmark_topsprof.py \
        --python-args "--num-processes 8 --total-tokens 64 --hidden 7168 --num-topk 8 --num-experts 256 --imbalance-ratio 40"

    # Parse-only mode (skip execution, parse existing csvs from all nodes)
    python3 run_topsprof.py --parse-only \
        --python-args "--num-processes 8 --total-tokens 64 --hidden 7168 --num-topk 8 --num-experts 256 --imbalance-ratio 40"

    # Specify csv/json paths explicitly (single csv)
    python3 run_topsprof.py --csv trace.csv --output-json result.json --parse-only \
        --python-args "--num-processes 8 --num-tokens 8 --hidden 7168 --num-topk 8 --num-experts 256"
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

num_nodes = int(os.getenv('WORLD_SIZE', 1))
node_rank = int(os.getenv('RANK', 0))


@dataclass
class KernelTrace:
    trace_type: str
    start: float      # in microseconds
    duration: float   # in microseconds
    kernel_id: int
    device_id: int
    stream_id: int
    name: str


def parse_time_to_us(time_str: str) -> float:
    """Parse time string (e.g., '193.14ms', '109.52us', '0.00ns') to microseconds."""
    time_str = time_str.strip()
    match = re.match(r'([\d.]+)(ns|us|ms|s)', time_str)
    if not match:
        raise ValueError(f"Cannot parse time string: '{time_str}'")
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {'ns': 0.001, 'us': 1.0, 'ms': 1000.0, 's': 1_000_000.0}
    return value * multipliers[unit]


def extract_kernel_type(full_name: str) -> str:
    """Extract kernel type ('dispatch' or 'combine') from the full kernel name."""
    if '::dispatch' in full_name:
        return 'dispatch'
    elif '::combine' in full_name:
        return 'combine'
    match = re.match(r'void\s+([\w:]+)[<(]', full_name)
    if match:
        return match.group(1).split('::')[-1]
    return full_name.split('(')[0].strip().split('::')[-1]


def parse_csv(csv_path: str) -> List[KernelTrace]:
    """Parse topsprof CSV trace file into a list of KernelTrace objects."""
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return []

    traces = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Warning: CSV file is empty: {csv_path}", file=sys.stderr)
            return []
        expected_header = ['Type', 'Start', 'Duration', 'KernelId', 'DeviceId', 'StreamId', 'Name']
        if header != expected_header:
            print(f"Warning: CSV header mismatch. Expected {expected_header}, got {header}", file=sys.stderr)

        for row in reader:
            if len(row) < 7:
                continue
            try:
                trace = KernelTrace(
                    trace_type=row[0],
                    start=parse_time_to_us(row[1]),
                    duration=parse_time_to_us(row[2]),
                    kernel_id=int(row[3]),
                    device_id=int(row[4]),
                    stream_id=int(row[5]),
                    name=row[6],
                )
                traces.append(trace)
            except (ValueError, IndexError) as e:
                print(f"Warning: skipping malformed row: {row} ({e})", file=sys.stderr)

    return traces


def analyse_multi_node_kernel_durations(all_node_traces: List[List[KernelTrace]],
                                         kernel_names: List[str],
                                         skip_first: int = 1,
                                         verbose: bool = False) -> dict:
    """
    Analyze kernel durations by merging traces from all nodes.

    JSON performance result calculation:
      1. Merge all traces from all nodes' CSVs into one combined list
      2. Group by kernel type (dispatch/combine) from the Name field
      3. Group by KernelId (same KernelId across nodes = same collective call across all ranks)
      4. For each KernelId, take MAX duration across ALL devices from ALL nodes
         (a collective completes only when the slowest device finishes)
      5. Sort KernelIds chronologically, skip first N instances as warmup
      6. Compute avg/min/max of the remaining max-duration values

    Output JSON format:
      [{"dispatch": {"latency_us": avg, "latency_us_min": min, "latency_us_max": max},
        "combine": {...}}]
    """
    # Step 1: Merge all node traces
    merged_traces = []
    for traces in all_node_traces:
        merged_traces.extend(traces)

    kernel_results = {}
    for kernel_name in kernel_names:
        # Step 2: Filter by kernel type
        kernel_traces = [t for t in merged_traces if extract_kernel_type(t.name) == kernel_name]
        if not kernel_traces:
            print(f"Warning: kernel '{kernel_name}' not found in traces", file=sys.stderr)
            continue

        # Step 3: Group by KernelId
        by_kernel_id: dict[int, List[KernelTrace]] = defaultdict(list)
        for t in kernel_traces:
            by_kernel_id[t.kernel_id].append(t)

        # Step 4: Sort KernelIds chronologically, take max across all devices
        sorted_kernel_ids = sorted(by_kernel_id.keys(),
                                   key=lambda kid: min(t.start for t in by_kernel_id[kid]))

        all_max_durations = []
        for kid in sorted_kernel_ids:
            device_durations = [t.duration for t in by_kernel_id[kid]]
            max_dur = max(device_durations)
            all_max_durations.append(max_dur)
            if verbose:
                print(f"  Kernel {kernel_name}, KernelId {kid}, "
                      f"devices={len(device_durations)}, max={max_dur:.2f} us")

        # Step 5: Skip warmup
        if skip_first >= len(all_max_durations):
            print(f"Warning: skip_first={skip_first} >= total instances={len(all_max_durations)} "
                  f"for kernel '{kernel_name}', using all instances", file=sys.stderr)
            effective_durations = all_max_durations
        else:
            effective_durations = all_max_durations[skip_first:]

        if not effective_durations:
            print(f"Warning: no valid instances for kernel '{kernel_name}'", file=sys.stderr)
            continue

        # Step 6: Compute statistics
        avg_us = statistics.mean(effective_durations)
        min_us = min(effective_durations)
        max_us = max(effective_durations)

        kernel_results[kernel_name] = {
            'latency_us': avg_us,
            'latency_us_min': min_us,
            'latency_us_max': max_us,
        }

        if verbose:
            print(f"  => {kernel_name}: nodes={len(all_max_durations)}, instances={len(effective_durations)}, "
                  f"avg={avg_us:.3f} us, min={min_us:.3f} us, max={max_us:.3f} us")

    return kernel_results


def print_performance_results(results: dict, num_nodes: int = 1):
    """Print formatted performance results."""
    print("\n" + "=" * 70)
    print(f"PERFORMANCE RESULTS (merged from {num_nodes} node(s))")
    print("=" * 70)
    print(f"{'Kernel':<12} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12}")
    print(f"{'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")
    for kernel_name, stats in results.items():
        print(f"{kernel_name:<12} {stats['latency_us']:>12.3f} "
              f"{stats['latency_us_min']:>12.3f} {stats['latency_us_max']:>12.3f}")
    print("=" * 70)


def build_topsprof_command(csv_path: str, python_cmd: str, python_script: str,
                           python_args: str, topsprof_args: str) -> str:
    """Build the full topsprof command string."""
    default_topsprof_opts = (
        "--force-overwrite "
        f"--export-csv {csv_path} "
        "--print-gcu-trace "
        "--enable-activities operator "
        "--print-app-log "
        "--trace topstx,runtime,torch_gcu,TopsFlame "
        "--topstx-domain-include topsrt,TE_GCU"
    )

    if topsprof_args:
        topsprof_opts = f"--force-overwrite --export-csv {csv_path} --print-gcu-trace {topsprof_args}"
    else:
        topsprof_opts = default_topsprof_opts

    cmd = f"topsprof {topsprof_opts} {python_cmd} {python_script}"
    if python_args:
        cmd += f" {python_args}"
    return cmd


def run_topsprof(cmd: str) -> int:
    """Execute topsprof command and return exit code."""
    print(f"\nExecuting command:\n  {cmd}\n")
    print("-" * 70)
    result = subprocess.run(cmd, shell=True)
    print("-" * 70)
    print(f"\nCommand exited with code: {result.returncode}")
    return result.returncode


def parse_python_args_for_naming(python_args: str) -> dict:
    """Parse --python-args string to extract parameters for filename generation."""
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument('--num-processes', type=int, default=8)
    pa.add_argument('--num-tokens', type=int, default=128)
    pa.add_argument('--hidden', type=int, default=7168)
    pa.add_argument('--num-topk', type=int, default=8)
    pa.add_argument('--num-experts', type=int, default=256)
    pa.add_argument('--total-tokens', type=int, default=None)
    pa.add_argument('--imbalance-ratio', type=int, default=0)
    parsed, _ = pa.parse_known_args(python_args.split())
    return vars(parsed)


def generate_csv_basename(python_args: str) -> str:
    """Generate CSV file basename (with node prefix) from --python-args.
    Format: node{node}_ranks{ranks}_tokens{tokens}_hidden{hidden}_experts{experts}_topk{topk}_ibratio{ibratio}
    """
    p = parse_python_args_for_naming(python_args)
    num_ranks = p['num_processes'] * num_nodes
    total_tokens = p['total_tokens'] if p['total_tokens'] is not None else p['num_tokens'] * num_ranks
    imbalance_ratio = p['imbalance_ratio']

    return (f"node{node_rank}_ranks{num_ranks}_tokens{total_tokens}"
            f"_hidden{p['hidden']}_experts{p['num_experts']}"
            f"_topk{p['num_topk']}_ibratio{imbalance_ratio}")


def generate_json_basename(python_args: str) -> str:
    """Generate JSON file basename (no node prefix, starts with ranks) from --python-args.
    Format: ranks{ranks}_tokens{tokens}_hidden{hidden}_experts{experts}_topk{topk}_ibratio{ibratio}
    """
    p = parse_python_args_for_naming(python_args)
    num_ranks = p['num_processes'] * num_nodes
    total_tokens = p['total_tokens'] if p['total_tokens'] is not None else p['num_tokens'] * num_ranks
    imbalance_ratio = p['imbalance_ratio']

    return (f"ranks{num_ranks}_tokens{total_tokens}"
            f"_hidden{p['hidden']}_experts{p['num_experts']}"
            f"_topk{p['num_topk']}_ibratio{imbalance_ratio}")


def discover_node_csvs(csv_path: str) -> List[str]:
    """Given a csv path like node0_ranks16_..., discover all node CSVs (node0_, node1_, ...).
    If the csv_path doesn't contain 'node' prefix, return it as-is.
    """
    dirname = os.path.dirname(csv_path) or '.'
    basename = os.path.basename(csv_path)

    match = re.match(r'node\d+_(.*)', basename)
    if not match:
        return [csv_path]

    suffix = match.group(1)
    pattern = os.path.join(dirname, f"node*_{suffix}")
    found = sorted(glob.glob(pattern))
    return found if found else [csv_path]


def main():
    parser = argparse.ArgumentParser(
        description='Run topsprof profiling and parse the resulting CSV trace.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to the output CSV file. If not specified, auto-generated from --python-args.')
    parser.add_argument('--python-script', type=str, default=None,
                        help='Python script to profile')
    parser.add_argument('--python-cmd', type=str, default='python3.12',
                        help='Python interpreter command (default: python3.12)')
    parser.add_argument('--python-args', type=str, default='',
                        help='Arguments to pass to the python script (quoted string)')
    parser.add_argument('--topsprof-args', type=str, default='',
                        help='Custom topsprof arguments (overrides defaults except --force-overwrite and --export-csv)')
    parser.add_argument('--parse-only', action='store_true',
                        help='Skip execution, only parse existing CSV file(s)')
    parser.add_argument('--no-parse', action='store_true',
                        help='Only run the command, skip CSV parsing')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output JSON file path. If not specified, auto-generated (ranks_... format).')
    parser.add_argument('--kernel-names', type=str, default='dispatch,combine',
                        help='Comma-separated kernel names to analyze (default: dispatch,combine)')
    parser.add_argument('--skip-first', type=int, default=50,
                        help='Number of initial kernel instances to skip as warmup (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-instance information')

    args = parser.parse_args()

    # Auto-generate filenames from --python-args if not specified
    if args.csv is None:
        args.csv = f"{generate_csv_basename(args.python_args)}.csv"
        print(f"Auto-generated CSV path: {args.csv}")

    if args.output_json is None:
        args.output_json = f"{generate_json_basename(args.python_args)}.json"
        print(f"Auto-generated JSON path: {args.output_json}")

    # Run topsprof command
    if not args.parse_only:
        if not args.python_script:
            parser.error("--python-script is required when not using --parse-only")

        cmd = build_topsprof_command(
            csv_path=args.csv,
            python_cmd=args.python_cmd,
            python_script=args.python_script,
            python_args=args.python_args,
            topsprof_args=args.topsprof_args,
        )
        exit_code = run_topsprof(cmd)
        if exit_code != 0:
            print(f"\nWarning: topsprof exited with non-zero code {exit_code}", file=sys.stderr)

    # Parse CSV(s) and compute results (only on node 0)
    if not args.no_parse and node_rank == 0:
        # Wait for all node CSVs to be ready (non-empty)
        expected_csv_count = num_nodes
        deadline = time.time() + 300
        while True:
            csv_files = discover_node_csvs(args.csv)
            csv_files = [f for f in csv_files if os.path.isfile(f)]
            ready_files = [f for f in csv_files if os.path.getsize(f) > 0]
            if len(ready_files) >= expected_csv_count:
                csv_files = ready_files
                break
            if time.time() > deadline:
                print(f"Warning: timeout waiting for {expected_csv_count} CSVs, "
                      f"found {len(ready_files)} ready. Proceeding with available files.",
                      file=sys.stderr)
                csv_files = ready_files
                break
            remaining = int(deadline - time.time())
            print(f"Waiting for all node CSVs... ({len(ready_files)}/{expected_csv_count} ready, "
                  f"timeout in {remaining}s)", flush=True)
            time.sleep(5)

        if not csv_files:
            print(f"Error: no CSV files found matching pattern from: {args.csv}", file=sys.stderr)
            sys.exit(1)

        print(f"\nParsing {len(csv_files)} CSV file(s): {csv_files}")

        # Parse all node CSVs
        all_node_traces = []
        for csv_file in csv_files:
            traces = parse_csv(csv_file)
            if traces:
                all_node_traces.append(traces)
                print(f"  {csv_file}: {len(traces)} traces")
            else:
                print(f"  {csv_file}: no traces (skipped)", file=sys.stderr)

        if not all_node_traces:
            print("Error: no valid traces found in any CSV file.", file=sys.stderr)
            sys.exit(1)

        kernel_names = [k.strip() for k in args.kernel_names.split(',')]
        results = analyse_multi_node_kernel_durations(
            all_node_traces, kernel_names,
            skip_first=args.skip_first,
            verbose=args.verbose
        )

        if not results:
            print("No kernel results computed.", file=sys.stderr)
            sys.exit(1)

        # Print results
        print_performance_results(results, num_nodes=len(all_node_traces))

        # Output JSON
        json_output = [results]
        with open(args.output_json, 'w') as f:
            json.dump(json_output, f, indent=4)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == '__main__':
    main()
