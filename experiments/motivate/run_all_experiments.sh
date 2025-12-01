#!/bin/bash
# Run All Motivation Experiments
# Usage: bash experiments/motivate/run_all_experiments.sh [GPU1] [GPU2] [GPU3] [GPU4]
#   If no arguments provided, defaults to GPUs 0, 1, 2, 3

# Parse GPU IDs (default to 0, 1, 2, 3 if not provided)
GPU1=${1:-0}
GPU2=${2:-1}
GPU3=${3:-2}
GPU4=${4:-3}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
LOG_DIR="./results/motivation/logs"
mkdir -p "$LOG_DIR"

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Running All Motivation Experiments (Parallel)"
echo "=========================================="
echo "GPU Assignment:"
echo "  - Exp 1: GPU $GPU1"
echo "  - Exp 2: GPU $GPU2"
echo "  - Exp 3: GPU $GPU3"
echo "  - Exp 4: GPU $GPU4"
echo "Log Directory: $LOG_DIR"
echo "=========================================="
echo ""

# Start experiments 1-4 in parallel
echo "Starting experiments 1-4 in parallel..."
echo ""

# Experiment 1: Latency Distribution
LOG1="$LOG_DIR/exp1_${TIMESTAMP}.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Exp 1 on GPU $GPU1 (log: $(basename $LOG1))"
(
    echo "=========================================="
    echo "Experiment 1: Latency Distribution"
    echo "GPU: $GPU1"
    echo "Started at: $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_exp1.sh" "$GPU1"
    echo ""
    echo "Experiment 1 completed at: $(date)"
) > "$LOG1" 2>&1 &
PID1=$!

# Experiment 2: Component Profiling
LOG2="$LOG_DIR/exp2_${TIMESTAMP}.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Exp 2 on GPU $GPU2 (log: $(basename $LOG2))"
(
    echo "=========================================="
    echo "Experiment 2: Component Profiling"
    echo "GPU: $GPU2"
    echo "Started at: $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_exp2.sh" "$GPU2"
    echo ""
    echo "Experiment 2 completed at: $(date)"
) > "$LOG2" 2>&1 &
PID2=$!

# Experiment 3: Vision Tokens vs Latency
LOG3="$LOG_DIR/exp3_${TIMESTAMP}.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Exp 3 on GPU $GPU3 (log: $(basename $LOG3))"
(
    echo "=========================================="
    echo "Experiment 3: Vision Tokens vs Latency"
    echo "GPU: $GPU3"
    echo "Started at: $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_exp3.sh" "$GPU3"
    echo ""
    echo "Experiment 3 completed at: $(date)"
) > "$LOG3" 2>&1 &
PID3=$!

# Experiment 4: Language Tokens vs Latency
LOG4="$LOG_DIR/exp4_${TIMESTAMP}.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Exp 4 on GPU $GPU4 (log: $(basename $LOG4))"
(
    echo "=========================================="
    echo "Experiment 4: Language Tokens vs Latency"
    echo "GPU: $GPU4"
    echo "Started at: $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_exp4.sh" "$GPU4"
    echo ""
    echo "Experiment 4 completed at: $(date)"
) > "$LOG4" 2>&1 &
PID4=$!

# Wait for all parallel experiments to complete
echo ""
echo "Waiting for experiments 1-4 to complete..."
echo "You can monitor progress by tailing the log files:"
echo "  tail -f $LOG1"
echo "  tail -f $LOG2"
echo "  tail -f $LOG3"
echo "  tail -f $LOG4"
echo ""

wait $PID1
STATUS1=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Exp 1 finished (exit code: $STATUS1)"

wait $PID2
STATUS2=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Exp 2 finished (exit code: $STATUS2)"

wait $PID3
STATUS3=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Exp 3 finished (exit code: $STATUS3)"

wait $PID4
STATUS4=$?
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Exp 4 finished (exit code: $STATUS4)"

echo ""
echo "=========================================="
echo "Experiments 1-4 Completed"
echo "=========================================="
echo "Exit codes: Exp1=$STATUS1, Exp2=$STATUS2, Exp3=$STATUS3, Exp4=$STATUS4"
echo ""

# Check if any experiment failed
if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ] || [ $STATUS3 -ne 0 ] || [ $STATUS4 -ne 0 ]; then
    echo "WARNING: Some experiments failed! Check the log files for details."
    echo ""
fi

# Experiment 5: FLOPs vs Latency (requires Exp 3 and 4 results)
echo "=========================================="
echo "Experiment 5: FLOPs vs Latency"
echo "=========================================="
LOG5="$LOG_DIR/exp5_${TIMESTAMP}.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Exp 5 (log: $(basename $LOG5))"
(
    echo "=========================================="
    echo "Experiment 5: FLOPs vs Latency"
    echo "Started at: $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/run_exp5.sh"
    echo ""
    echo "Experiment 5 completed at: $(date)"
) > "$LOG5" 2>&1
STATUS5=$?

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Results saved in: ./results/motivation"
echo "  - Exp 1: ./results/motivation/exp1"
echo "  - Exp 2: ./results/motivation/exp2"
echo "  - Exp 3: ./results/motivation/exp3"
echo "  - Exp 4: ./results/motivation/exp4"
echo "  - Exp 5: ./results/motivation/exp5"
echo ""
echo "Log files saved in: $LOG_DIR"
echo "  - Exp 1: $(basename $LOG1)"
echo "  - Exp 2: $(basename $LOG2)"
echo "  - Exp 3: $(basename $LOG3)"
echo "  - Exp 4: $(basename $LOG4)"
echo "  - Exp 5: $(basename $LOG5)"
echo ""
echo "Exit codes: Exp1=$STATUS1, Exp2=$STATUS2, Exp3=$STATUS3, Exp4=$STATUS4, Exp5=$STATUS5"
echo "=========================================="

