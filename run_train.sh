#!/bin/bash
#SBATCH --job-name=ct2rep_radgenome
#SBATCH --output=/umbc/rs/pi_oates/users/dta1/trained_model/Medical_Report_Generation/logs/ct2rep_%j.out
#SBATCH --error=/umbc/rs/pi_oates/users/dta1/trained_model/Medical_Report_Generation/logs/ct2rep_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

# ============================================================================
# CT2Rep Training on RadGenome-ChestCT Dataset
# UMBC Cluster - 4x NVIDIA L40S GPUs
# ============================================================================

echo "========================================"
echo "CT2Rep Training on RadGenome-ChestCT"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Environment setup
module load cuda/13.1
module load python/3.10

# Activate virtual environment
source /umbc/ada/oates/users/dta1/venv/bin/activate

# Set paths
export DATA_DIR="/umbc/rs/pi_oates/users/dta1/Data/Medical_Report_Generation/RadGenome_dataset"
export OUTPUT_DIR="/umbc/rs/pi_oates/users/dta1/trained_model/Medical_Report_Generation"
export CACHE_DIR="/umbc/rs/pi_oates/users/dta1/all_cache"

# HuggingFace cache
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"

# NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Create output directories
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/results

# Change to code directory
cd /path/to/CT2Rep

# Print environment info
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Cache directory: ${CACHE_DIR}"
echo ""

# Check if data exists
echo "Checking data availability..."
ls -la ${DATA_DIR}/dataset/radgenome_files/ 2>/dev/null || echo "radgenome_files not found"
ls -la ${DATA_DIR}/dataset/train_preprocessed/ 2>/dev/null | head -5 || echo "train_preprocessed not found"
echo ""

# Run training
python main.py \
    --data_dir ${DATA_DIR} \
    --n_gpu 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 30 \
    --lr 5e-5 \
    --max_seq_length 300 \
    --threshold 3 \
    --use_amp \
    --amp_dtype float16 \
    --eval_interval 5 \
    --log_interval 10 \
    --save_interval 5 \
    --num_inference_samples 5 \
    --beam_size 3

echo ""
echo "========================================"
echo "Training completed at $(date)"
echo "========================================"