#!/bin/bash

# Constants
NUM_GPUS=8           # Number of GPUs to use
BATCH_SIZE=25         # Batch size for training
APP_NUM_NODES=1       # Number of application nodes
DATA_PATH="/home/mansour/ML3300-24a/omersela3/tensors-v5"  # Data path
CPU_WORKERS=12        # Number of CPU workers
ACCELERATOR="gpu"     # Type of accelerator
STRATEGY="ddp"        # Strategy to use
OVERWRITE_CONFIG_PATH="src/config/cnn_skip_connection/config-v5-msl-1979-2024-24d-lookback3h-forecast4h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8.ini"
MODE="EVAL,EVAL_STORM_CLASSIFICATION"

# Print environment details for debugging
echo "Starting job on $(hostname) at $(date)"
echo "Running in directory $(pwd)"
echo "Using the following Python executable: $(which python)"
echo "Using the following conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"

# Activate the conda environment
bash
eval "$(conda shell.bash hook)"
conda activate base
export MPLCONFIGDIR=/home/mansour/ML3300-24a/omersela3/matplotlib_cache

# Print the arguments to be passed to the Python script
echo "Running script with the following arguments:"
echo "  --APP_overwrite_config_path=$OVERWRITE_CONFIG_PATH"
echo "  --APP_mode=$MODE"
echo "  --APP_num_nodes=$APP_NUM_NODES"
echo "  --DATA_path=$DATA_PATH"
echo "  --APP_cpu_workers=$CPU_WORKERS"
echo "  --APP_accelerator=$ACCELERATOR"
echo "  --APP_strategy=$STRATEGY"
echo "  --APP_devices=$NUM_GPUS"
echo "  --APP_batch_size=$BATCH_SIZE"

# Run the Python script with the defined constants
python __init__.py \
  --APP_overwrite_config_path=$OVERWRITE_CONFIG_PATH \
  --APP_mode=$MODE \
  --APP_num_nodes=$APP_NUM_NODES \
  --DATA_path=$DATA_PATH \
  --APP_cpu_workers=$CPU_WORKERS \
  --APP_accelerator=$ACCELERATOR \
  --APP_strategy=$STRATEGY \
  --APP_devices=$NUM_GPUS \
  --APP_batch_size=$BATCH_SIZE
