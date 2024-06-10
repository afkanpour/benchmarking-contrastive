#!/bin/bash

#SBATCH --job-name=multimodal_project
#SBATCH --partition=rtx6000,t4v1,t4v2
#SBATCH --mem-per-gpu=10GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output=../outputs/slurm-%j-%N.out
#SBATCH --open-mode=append

# load virtual environment
source ~/Documents/envs/mmm/bin/activate

cd ~/Documents/GitHub/Multimodal_val_loops/MedMultiModal/src/
git checkout val-loops
export PYTHONPATH="."

export NCCL_IB_DISABLE=1  # disable InfiniBand (the Vector cluster does not have it)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export NCCL_ASYNC_ERROR_HANDLING=1 # set to 1 for NCCL backend
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=12

export MASTER_ADDR=$(hostname)
export MASTER_PORT=45678

nvidia-smi
echo SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}
echo SLURM_JOBID=${SLURM_JOBID}

# “srun” executes the script <ntasks-per-node * nodes> times
srun --export=ALL -N $SLURM_JOB_NUM_NODES --cpu_bind=v --accel-bind=gn python -u training/validate.py \
    --model ViT-B-32 \
    --checkpoints-dir /projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_64/checkpoints/ \
    --val-data /projects/multimodal/datasets/pmc_oa/valid.jsonl  \
    --dataset-type mixed \
    --csv-separator , \
    --val-no-retrieval \
    --batch-size 64 \
    --accum-freq 4 \
    --workers 4 \
    --aug-cfg quilt_crop=True \
    --wd 0.1 \
    --name ValLoss_ViT_B_32_Batchsize_64 \
    --resume latest \
    --gather-with-grad \
    --logs /checkpoint/$USER/$SLURM_JOBID/ \
    --zeroshot-frequency 1 \
    --report-to wandb
