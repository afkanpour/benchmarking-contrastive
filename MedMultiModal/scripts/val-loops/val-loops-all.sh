#!/bin/bash

#SBATCH --job-name=multimodal_project
#SBATCH --partition=a100
#SBATCH --qos=a100_arashaf
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
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


# CHECKPOINTS
# batch size = 8
BS8_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_8/checkpoints/
# batch size = 16
BS16_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_16/checkpoints/
# batch size = 32
BS32_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_32/checkpoints/
# batch size = 64
BS64_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_64/checkpoints/
# batch size = 128
BS128_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_128/checkpoints/
# batch size = 256
BS256_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_256/checkpoints/
# batch size = 512
BS512_CKPT=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_512/checkpoints/
# batch size = 1024
BS1024_CKPT=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-2/checkpoints/

i=8
for CKPT in $BS8_CKPT $BS16_CKPT $BS32_CKPT $BS64_CKPT $BS128_CKPT $BS256_CKPT $BS512_CKPT $BS1024_CKPT 
do  
    echo RUNNING BATCH SIZE $i: $CKPT

    # “srun” executes the script <ntasks-per-node * nodes> times
    srun --export=ALL -N $SLURM_JOB_NUM_NODES --cpu_bind=v --accel-bind=gn python -u training/validate.py \
        --model ViT-B-32 \
        --checkpoints-dir $CKPT \
        --val-data /projects/multimodal/datasets/pmc_oa/valid.jsonl  \
        --dataset-type mixed \
        --csv-separator , \
        --val-no-retrieval \
        --batch-size 32 \
        --accum-freq 4 \
        --workers 4 \
        --aug-cfg quilt_crop=True \
        --wd 0.1 \
        --name ValLoss_ViT_B_32_Batchsize_$i \
        --resume latest \
        --gather-with-grad \
        --logs /checkpoint/$USER/$SLURM_JOBID/ \
        --zeroshot-frequency 1 \
        --report-to wandb

    i=$(($i*2))
done
