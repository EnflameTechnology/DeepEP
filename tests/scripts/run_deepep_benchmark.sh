#!/bin/bash

# Script to run benchmark tests with different token numbers and collect performance results
# Supports: test_low_latency_benchmark.py, test_intranode_benchmark.py, test_internode_benchmark.py
#
# Usage: ./run_deepep_benchmark.sh [options]
#
# Options:
#   -s, --script <path>       Path to the test script (default: test_low_latency_benchmark.py)
#   -p, --num-processes <n>   Number of processes (default: 8)
#   -d, --hidden <n>          Hidden dimension size (default: 7168)
#   -k, --num-topk <n>        Number of top-k experts (default: 8)
#   -e, --num-experts <n>     Number of experts (default: 256)
#   -t, --tokens <list>       Comma-separated list of token numbers (default: 2,4,8,16,32,48,64,80,96,112,128,256)
#   --use-kineto              Use kineto profiling (for intranode/internode benchmarks)
#   -h, --help                Show this help message
#
# Examples:
#   ./run_deepep_benchmark.sh
#   ./run_deepep_benchmark.sh -p 4 -d 4096
#   ./run_deepep_benchmark.sh -s test_low_latency_benchmark.py -e 288
#   ./run_deepep_benchmark.sh -s test_intranode_benchmark.py -t 1024,2048,4096
#   ./run_deepep_benchmark.sh -s test_internode_benchmark.py --use-kineto

set -e
set -o pipefail  # Ensure pipeline errors are caught

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters
TEST_SCRIPT="$SCRIPT_DIR/../test_low_latency_benchmark.py"
NUM_PROCESSES=8
HIDDEN=7168
NUM_TOPK=8
NUM_EXPERTS=256
TOKEN_NUMS_STR="2,4,8,16,32,48,64,80,96,112,128,256"
USE_KINETO=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--script)
            TEST_SCRIPT="$2"
            # If not absolute path, prepend SCRIPT_DIR/../ (tests directory)
            if [[ "$TEST_SCRIPT" != /* ]]; then
                TEST_SCRIPT="$SCRIPT_DIR/../$TEST_SCRIPT"
            fi
            shift 2
            ;;
        -p|--num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -d|--hidden)
            HIDDEN="$2"
            shift 2
            ;;
        -k|--num-topk)
            NUM_TOPK="$2"
            shift 2
            ;;
        -e|--num-experts)
            NUM_EXPERTS="$2"
            shift 2
            ;;
        -t|--tokens)
            TOKEN_NUMS_STR="$2"
            shift 2
            ;;
        --use-kineto)
            USE_KINETO=true
            shift
            ;;
        -h|--help)
            head -n 24 "$0" | tail -n +3
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Verify test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

# Convert comma-separated token numbers to array
IFS=',' read -ra TOKEN_NUMS <<< "$TOKEN_NUMS_STR"

# Detect script type based on filename
SCRIPT_BASENAME=$(basename "$TEST_SCRIPT")
SCRIPT_TYPE=""
if [[ "$SCRIPT_BASENAME" == *"low_latency"* ]]; then
    SCRIPT_TYPE="low_latency"
elif [[ "$SCRIPT_BASENAME" == *"intranode"* ]]; then
    SCRIPT_TYPE="intranode"
elif [[ "$SCRIPT_BASENAME" == *"internode"* ]]; then
    SCRIPT_TYPE="internode"
else
    echo "Warning: Unknown script type, defaulting to low_latency format"
    SCRIPT_TYPE="low_latency"
fi

# Output directory for results
RESULTS_DIR="./token_sweep_results_${SCRIPT_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
CSV_FILE="$RESULTS_DIR/results.csv"

echo "=========================================="
echo "Benchmark Token Sweep Test"
echo "=========================================="
echo "Test script: $TEST_SCRIPT"
echo "Script type: $SCRIPT_TYPE"
echo "Num processes: $NUM_PROCESSES"
echo "Hidden: $HIDDEN"
echo "Num topk: $NUM_TOPK"
echo "Num experts: $NUM_EXPERTS"
echo "Token numbers: ${TOKEN_NUMS[*]}"
echo "Use kineto: $USE_KINETO"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Initialize CSV file with header based on script type
case "$SCRIPT_TYPE" in
    "low_latency")
        echo "token_num,dispatch_bw,dispatch_latency,combine_bw,combine_latency" > "$CSV_FILE"
        ;;
    "intranode")
        echo "token_num,bf16_dispatch_bw,bf16_dispatch_latency,bf16_combine_bw,bf16_combine_latency,fp8_dispatch_bw,fp8_dispatch_latency,fp8_combine_bw,fp8_combine_latency" > "$CSV_FILE"
        ;;
    "internode")
        echo "token_num,bf16_dispatch_rdma_bw,bf16_dispatch_nvl_bw,bf16_dispatch_latency,bf16_combine_rdma_bw,bf16_combine_nvl_bw,bf16_combine_latency,fp8_dispatch_rdma_bw,fp8_dispatch_nvl_bw,fp8_dispatch_latency,fp8_combine_rdma_bw,fp8_combine_nvl_bw,fp8_combine_latency" > "$CSV_FILE"
        ;;
esac

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
========================================
Benchmark Token Sweep Test Results
========================================
Configuration:
  - Test script: $TEST_SCRIPT
  - Script type: $SCRIPT_TYPE
  - Num processes: $NUM_PROCESSES
  - Hidden dimension: $HIDDEN
  - Num topk: $NUM_TOPK
  - Num experts: $NUM_EXPERTS
  - Use kineto: $USE_KINETO
  - Test date: $(date)

========================================
Results (Dispatch & Combine):
========================================

EOF

# Function to extract performance metrics for low_latency script
extract_metrics_low_latency() {
    local token_num=$1
    local log_file=$2
    
    # Log file existence is already checked in extract_metrics()
    if [ ! -f "$log_file" ]; then
        return 0
    fi
    
    # Extract separate profiling statistics (Dispatch and Combine)
    # Format: [avg across N ranks] Dispatch bandwidth: XXX GB/s, avg_t=XXX us | Combine bandwidth: XXX GB/s, avg_t=XXX us
    local separate_line=$(grep "\[avg across.*ranks\] Dispatch bandwidth:" "$log_file" 2>/dev/null | tail -1)
    if [ -z "$separate_line" ]; then
        # Fall back to rank 0 results
        separate_line=$(grep "\[rank 0\] Dispatch bandwidth:" "$log_file" 2>/dev/null | tail -1)
    fi
    
    local dispatch_bw=""
    local dispatch_t=""
    local combine_bw=""
    local combine_t=""
    
    if [ -n "$separate_line" ]; then
        dispatch_bw=$(echo "$separate_line" | sed -n 's/.*Dispatch bandwidth: \([0-9.]*\) GB\/s.*/\1/p')
        dispatch_t=$(echo "$separate_line" | sed -n 's/.*Dispatch bandwidth:[^|]*avg_t=\([0-9.]*\) us.*/\1/p')
        combine_bw=$(echo "$separate_line" | sed -n 's/.*Combine bandwidth: \([0-9.]*\) GB\/s.*/\1/p')
        combine_t=$(echo "$separate_line" | sed -n 's/.*Combine bandwidth:.*avg_t=\([0-9.]*\) us.*/\1/p')
    fi
    
    # Handle empty values
    dispatch_bw=${dispatch_bw:-""}
    dispatch_t=${dispatch_t:-""}
    combine_bw=${combine_bw:-""}
    combine_t=${combine_t:-""}
    
    # Write to CSV
    echo "$token_num,$dispatch_bw,$dispatch_t,$combine_bw,$combine_t" >> "$CSV_FILE"
    
    # Write to summary
    cat >> "$SUMMARY_FILE" << EOF
Token Number: $token_num
----------------------------------------
  Dispatch: bandwidth=$dispatch_bw GB/s, latency=$dispatch_t us
  Combine:  bandwidth=$combine_bw GB/s, latency=$combine_t us

EOF
}

# Function to extract performance metrics for intranode script
extract_metrics_intranode() {
    local token_num=$1
    local log_file=$2
    
    # Log file existence is already checked in extract_metrics()
    if [ ! -f "$log_file" ]; then
        return 0
    fi
    
    # Initialize variables for both FP8 and BF16
    local fp8_dispatch_bw="" fp8_dispatch_t="" fp8_combine_bw="" fp8_combine_t=""
    local bf16_dispatch_bw="" bf16_dispatch_t="" bf16_combine_bw="" bf16_combine_t=""
    
    # Extract for each data type (FP8 and BF16)
    # Format: [avg across N ranks] (FP8/BF16) Dispatch[kineto] bandwidth: XXX GB/s, avg_t=XXX us | Combine[kineto] bandwidth: XXX GB/s, avg_t=XXX us
    for data_type in "FP8" "BF16"; do
        local separate_line=$(grep "\[avg across.*ranks\] ($data_type)" "$log_file" 2>/dev/null | tail -1)
        if [ -z "$separate_line" ]; then
            # Fall back to rank 0 results
            separate_line=$(grep "\[rank 0\] ($data_type)" "$log_file" 2>/dev/null | tail -1)
        fi
        
        local dispatch_bw=""
        local dispatch_t=""
        local combine_bw=""
        local combine_t=""
        
        if [ -n "$separate_line" ]; then
            dispatch_bw=$(echo "$separate_line" | sed -n 's/.*Dispatch[^:]*bandwidth: \([0-9.]*\) GB\/s.*/\1/p')
            dispatch_t=$(echo "$separate_line" | sed -n 's/.*Dispatch[^|]*avg_t=\([0-9.]*\) us.*/\1/p')
            combine_bw=$(echo "$separate_line" | sed -n 's/.*Combine[^:]*bandwidth: \([0-9.]*\) GB\/s.*/\1/p')
            combine_t=$(echo "$separate_line" | sed -n 's/.*Combine[^:]*bandwidth:.*avg_t=\([0-9.]*\) us.*/\1/p')
        fi
        
        # Store values based on data type
        if [ "$data_type" = "FP8" ]; then
            fp8_dispatch_bw=${dispatch_bw:-""}
            fp8_dispatch_t=${dispatch_t:-""}
            fp8_combine_bw=${combine_bw:-""}
            fp8_combine_t=${combine_t:-""}
        else
            bf16_dispatch_bw=${dispatch_bw:-""}
            bf16_dispatch_t=${dispatch_t:-""}
            bf16_combine_bw=${combine_bw:-""}
            bf16_combine_t=${combine_t:-""}
        fi
    done
    
    # Write to CSV (single row with BF16 and FP8 columns)
    echo "$token_num,$bf16_dispatch_bw,$bf16_dispatch_t,$bf16_combine_bw,$bf16_combine_t,$fp8_dispatch_bw,$fp8_dispatch_t,$fp8_combine_bw,$fp8_combine_t" >> "$CSV_FILE"
    
    # Write to summary
    cat >> "$SUMMARY_FILE" << EOF
Token Number: $token_num
----------------------------------------
  BF16:
    Dispatch: bandwidth=$bf16_dispatch_bw GB/s, latency=$bf16_dispatch_t us
    Combine:  bandwidth=$bf16_combine_bw GB/s, latency=$bf16_combine_t us
  FP8:
    Dispatch: bandwidth=$fp8_dispatch_bw GB/s, latency=$fp8_dispatch_t us
    Combine:  bandwidth=$fp8_combine_bw GB/s, latency=$fp8_combine_t us

EOF
}

# Function to extract performance metrics for internode script
extract_metrics_internode() {
    local token_num=$1
    local log_file=$2
    
    # Log file existence is already checked in extract_metrics()
    if [ ! -f "$log_file" ]; then
        return 0
    fi
    
    # Initialize variables for both FP8 and BF16
    local fp8_dispatch_rdma_bw="" fp8_dispatch_nvl_bw="" fp8_dispatch_t=""
    local fp8_combine_rdma_bw="" fp8_combine_nvl_bw="" fp8_combine_t=""
    local bf16_dispatch_rdma_bw="" bf16_dispatch_nvl_bw="" bf16_dispatch_t=""
    local bf16_combine_rdma_bw="" bf16_combine_nvl_bw="" bf16_combine_t=""
    
    # Extract for each data type (FP8 and BF16)
    # Format: [avg across N ranks] (FP8/BF16) Dispatch[kineto]: RDMA XXX GB/s, NVL XXX GB/s, t=XXX us | Combine[kineto]: RDMA XXX GB/s, NVL XXX GB/s, t=XXX us
    for data_type in "FP8" "BF16"; do
        local separate_line=$(grep "\[avg across.*ranks\] ($data_type)" "$log_file" 2>/dev/null | tail -1)
        if [ -z "$separate_line" ]; then
            # Fall back to rank 0 results
            separate_line=$(grep "\[rank 0\] ($data_type)" "$log_file" 2>/dev/null | tail -1)
        fi
        
        local dispatch_rdma_bw=""
        local dispatch_nvl_bw=""
        local dispatch_t=""
        local combine_rdma_bw=""
        local combine_nvl_bw=""
        local combine_t=""
        
        if [ -n "$separate_line" ]; then
            # Extract dispatch metrics
            dispatch_rdma_bw=$(echo "$separate_line" | sed -n 's/.*Dispatch[^:]*: RDMA \([0-9.]*\) GB\/s.*/\1/p')
            dispatch_nvl_bw=$(echo "$separate_line" | sed -n 's/.*Dispatch[^|]*NVL \([0-9.]*\) GB\/s.*/\1/p')
            dispatch_t=$(echo "$separate_line" | sed -n 's/.*Dispatch[^|]*t=\([0-9.]*\) us.*/\1/p')
            
            # Extract combine metrics (after the pipe)
            combine_rdma_bw=$(echo "$separate_line" | sed -n 's/.*| Combine[^:]*: RDMA \([0-9.]*\) GB\/s.*/\1/p')
            combine_nvl_bw=$(echo "$separate_line" | sed -n 's/.*| Combine[^:]*:.*NVL \([0-9.]*\) GB\/s.*/\1/p')
            combine_t=$(echo "$separate_line" | sed -n 's/.*| Combine[^:]*:.*t=\([0-9.]*\) us.*/\1/p')
        fi
        
        # Store values based on data type
        if [ "$data_type" = "FP8" ]; then
            fp8_dispatch_rdma_bw=${dispatch_rdma_bw:-""}
            fp8_dispatch_nvl_bw=${dispatch_nvl_bw:-""}
            fp8_dispatch_t=${dispatch_t:-""}
            fp8_combine_rdma_bw=${combine_rdma_bw:-""}
            fp8_combine_nvl_bw=${combine_nvl_bw:-""}
            fp8_combine_t=${combine_t:-""}
        else
            bf16_dispatch_rdma_bw=${dispatch_rdma_bw:-""}
            bf16_dispatch_nvl_bw=${dispatch_nvl_bw:-""}
            bf16_dispatch_t=${dispatch_t:-""}
            bf16_combine_rdma_bw=${combine_rdma_bw:-""}
            bf16_combine_nvl_bw=${combine_nvl_bw:-""}
            bf16_combine_t=${combine_t:-""}
        fi
    done
    
    # Write to CSV (single row with BF16 and FP8 columns)
    echo "$token_num,$bf16_dispatch_rdma_bw,$bf16_dispatch_nvl_bw,$bf16_dispatch_t,$bf16_combine_rdma_bw,$bf16_combine_nvl_bw,$bf16_combine_t,$fp8_dispatch_rdma_bw,$fp8_dispatch_nvl_bw,$fp8_dispatch_t,$fp8_combine_rdma_bw,$fp8_combine_nvl_bw,$fp8_combine_t" >> "$CSV_FILE"
    
    # Write to summary
    cat >> "$SUMMARY_FILE" << EOF
Token Number: $token_num
----------------------------------------
  BF16:
    Dispatch: RDMA=$bf16_dispatch_rdma_bw GB/s, NVL=$bf16_dispatch_nvl_bw GB/s, latency=$bf16_dispatch_t us
    Combine:  RDMA=$bf16_combine_rdma_bw GB/s, NVL=$bf16_combine_nvl_bw GB/s, latency=$bf16_combine_t us
  FP8:
    Dispatch: RDMA=$fp8_dispatch_rdma_bw GB/s, NVL=$fp8_dispatch_nvl_bw GB/s, latency=$fp8_dispatch_t us
    Combine:  RDMA=$fp8_combine_rdma_bw GB/s, NVL=$fp8_combine_nvl_bw GB/s, latency=$fp8_combine_t us

EOF
}

# Function to extract metrics based on script type
extract_metrics() {
    local token_num=$1
    local log_dir=$2
    
    # Determine log file path based on script type
    local log_file=""
    case "$SCRIPT_TYPE" in
        "low_latency")
            log_file="$log_dir/0_test_low_latency_perf.log"
            ;;
        "intranode"|"internode")
            log_file="$log_dir/run.log"
            ;;
    esac
    
    # Check if log file exists (may not exist on non-primary nodes in multi-node setup)
    if [ ! -f "$log_file" ]; then
        echo "Note: Log file not found: $log_file (this is normal on non-primary nodes)"
        echo "Skipping metric extraction for token_num=$token_num on this node"
        return 0
    fi
    
    # Call appropriate extraction function
    case "$SCRIPT_TYPE" in
        "low_latency")
            extract_metrics_low_latency "$token_num" "$log_file"
            ;;
        "intranode")
            extract_metrics_intranode "$token_num" "$log_file"
            ;;
        "internode")
            extract_metrics_internode "$token_num" "$log_file"
            ;;
    esac
}

# Function to write failed result to CSV based on script type
write_failed_csv() {
    local token_num=$1
    case "$SCRIPT_TYPE" in
        "low_latency")
            echo "$token_num,,,," >> "$CSV_FILE"
            ;;
        "intranode")
            # 8 empty columns: bf16_dispatch_bw,bf16_dispatch_latency,bf16_combine_bw,bf16_combine_latency,fp8_dispatch_bw,fp8_dispatch_latency,fp8_combine_bw,fp8_combine_latency
            echo "$token_num,,,,,,,," >> "$CSV_FILE"
            ;;
        "internode")
            # 12 empty columns: bf16_dispatch_rdma_bw,bf16_dispatch_nvl_bw,bf16_dispatch_latency,bf16_combine_rdma_bw,bf16_combine_nvl_bw,bf16_combine_latency,fp8_dispatch_rdma_bw,fp8_dispatch_nvl_bw,fp8_dispatch_latency,fp8_combine_rdma_bw,fp8_combine_nvl_bw,fp8_combine_latency
            echo "$token_num,,,,,,,,,,,," >> "$CSV_FILE"
            ;;
    esac
}

# Main loop: test each token number
for token_num in "${TOKEN_NUMS[@]}"; do
    echo "=========================================="
    echo "Testing with token_num=$token_num"
    echo "=========================================="
    
    # Create subdirectory for this run
    RUN_DIR="$RESULTS_DIR/token_$token_num"
    mkdir -p "$RUN_DIR"
    
    # Change to run directory
    cd "$RUN_DIR"
    
    # Build command based on script type
    CMD="python3 $TEST_SCRIPT --num-processes $NUM_PROCESSES --num-tokens $token_num --hidden $HIDDEN --num-topk $NUM_TOPK --num-experts $NUM_EXPERTS"
    
    # Add kineto flag if enabled (for intranode/internode)
    if [ "$USE_KINETO" = true ] && [ "$SCRIPT_TYPE" != "low_latency" ]; then
        CMD="$CMD --use-kineto"
    fi
    
    # Run the test
    echo "Running: $CMD"
    if eval "$CMD" 2>&1 | tee run.log; then
        
        echo "Test completed successfully!"
        
        # Extract metrics
        echo "Extracting performance metrics..."
        cd - > /dev/null
        extract_metrics "$token_num" "$RUN_DIR"
        
    else
        echo "Test FAILED for token_num=$token_num"
        cd - > /dev/null
        write_failed_csv "$token_num"
        echo -e "Token Number: $token_num\n----------------------------------------\nSTATUS: FAILED\n\n" >> "$SUMMARY_FILE"
    fi
    echo ""
    echo "Sleeping for 10 seconds..."
    sleep 10
    echo "Done sleeping"
done

# Generate final summary
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Results summary:"
cat "$SUMMARY_FILE"
echo ""
echo "CSV results saved to: $CSV_FILE"
echo "Full results directory: $RESULTS_DIR"
echo ""
