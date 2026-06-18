#!/bin/bash
# ==============================================================================
# Local Interactive Test Script - High-Performance K-Means
# Usage: ./scripts/local_test.sh [-t target] [-k clusters] [-j threads] [-d dataset]
# ==============================================================================

# ANSI Color Definitions
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Dynamically find the project root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

# Default values
TARGET="seq"
K=5
THREADS=4
DATASET_ARG="data/mall_customers_for_test.csv"

# ---------------------------------------------------------
# Smart Argument Parser (Named Flags)
# ---------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--target) TARGET="$2"; shift ;;
        -k|--clusters) K="$2"; shift ;;
        -j|--threads) THREADS="$2"; shift ;;
        -d|--dataset) DATASET_ARG="$2"; shift ;;
        -h|--help)
            echo -e "${CYAN}Usage: ./scripts/local_test.sh [OPTIONS]${NC}"
            echo "Options:"
            echo "  -t, --target    Algorithm version (seq, mpi, omp-gpu, cuda) [default: seq]"
            echo "  -k, --clusters  Number of clusters [default: 5]"
            echo "  -j, --threads   Number of OpenMP threads [default: 4]"
            echo "  -d, --dataset   Path to the dataset CSV [default: data/mall_customers.csv]"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown parameter passed: $1${NC}"
            echo "Use ./scripts/local_test.sh --help for usage."
            exit 1
            ;;
    esac
    shift
done

# Smart path resolution for the dataset
if [[ "$DATASET_ARG" = /* ]]; then
    DATASET="$DATASET_ARG"
else
    DATASET="$ROOT_DIR/$DATASET_ARG"
fi

# Inject hardware configuration via environment variable
export OMP_NUM_THREADS=$THREADS

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN} Local Test: K-Means ($TARGET) | K=$K | Threads=$THREADS${NC}"
echo -e "${CYAN} Dataset: $DATASET${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build/ directory not found. Please compile with CMake first.${NC}"
    exit 1
fi

# 2. Check if the dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}Error: Dataset not found at $DATASET.${NC}"
    exit 1
fi

# Helper function: Check if the C++ executable was actually built
check_executable() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: Executable '$1' not found.${NC}"
        echo -e "${YELLOW}Reason: This version might not be implemented yet, or compilation failed.${NC}"
        exit 1
    fi
}

# Helper function: Check if the machine has an NVIDIA GPU
check_nvidia_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Hardware Limitation:${NC}"
        echo -e "${RED}Error: No NVIDIA GPU detected on this machine.${NC}"
        echo "You can only run 'cuda' or 'omp-gpu' on a machine with NVIDIA drivers installed."
        exit 1
    fi
}

# 3. Router and Execution
case "$TARGET" in
    "seq")
        EXEC_PATH="$BUILD_DIR/kmeans_seq"
        check_executable "$EXEC_PATH"
        $EXEC_PATH -k "$K" -input "$DATASET"
        ;;
    "mpi")
        EXEC_PATH="$BUILD_DIR/kmeans_mpi_omp"
        check_executable "$EXEC_PATH"
        mpirun -np 2 "$EXEC_PATH" -k "$K" -input "$DATASET"
        ;;
    "omp-gpu")
        EXEC_PATH="$BUILD_DIR/kmeans_gpu"
        check_nvidia_gpu
        check_executable "$EXEC_PATH"
        "$EXEC_PATH" -k "$K" -input "$DATASET"
        ;;
    "cuda")
        EXEC_PATH="$BUILD_DIR/kmeans_cuda"
        check_nvidia_gpu
        check_executable "$EXEC_PATH"
        "$EXEC_PATH" -k "$K" -input "$DATASET"
        ;;
    *)
        echo -e "${RED}Error: Invalid target '$TARGET'.${NC}"
        echo "Valid options: seq, mpi, omp-gpu, cuda."
        exit 1
        ;;
esac
