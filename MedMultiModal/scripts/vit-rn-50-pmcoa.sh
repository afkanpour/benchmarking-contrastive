#!/bin/sh
#SBATCH --partition=compute_full_node                      # compute_full_node / compute
#SBATCH --nodes=1                               # number of nodes requested
#SBATCH --ntasks=1                              # this should be same as number of nodes
#SBATCH --gpus-per-node=4                       # 1/4
#SBATCH --output=output/vit-rn-50-pmc-oa.out
#SBATCH --open-mode=append                      # Append is important because otherwise preemption resets the file
# SBATCH --array=0-5%1                           # auto submit 2 times
#SBATCH --job-name=vit-rn-50-pmc-oa
#SBATCH --time=24:00:00

cd ../src
export PYTHONPATH="."

# load modules
module load MistEnv/2021a anaconda3
source activate pytorch_env

MASTER=$(/bin/hostname -s)
MPORT=$(shuf -i 6000-20000 -n 1)

echo ${SLURM_ARRAY_JOB_ID}
echo ${SLURM_JOBID}

srun bash -c "sleep \$(( SLURM_PROCID * 5 )); torchrun \
               --node_rank=\${SLURM_PROCID} \
               --nproc_per_node=4 \
               --nnodes=$SLURM_NNODES \
               --rdzv_backend=c10d \
               --rdzv_endpoint=$MASTER:$MPORT \
               --rdzv_id=$SLURM_JOB_ID \
               --max-restarts=10 \
               -m training.main \
                    --model RN50 \
                    --pretrained openai \
                    --train-data ./data/pmc_oa/train.jsonl --train-num-samples 1317219 --dataset-type mixed --csv-separator , \
                    --batch-size 256 \
                    --accum-freq 4 \
                    --workers 8 \
                    --lr 1e-5 \
                    --lr-scheduler const \
                    --epochs 15 \
                    --warmup 200 \
                    --aug-cfg quilt_crop=True \
                    --wd 0.1 \
                    --name ViT-RN-50-PMA-OA \
                    --resume latest \
                    --gather-with-grad \
                    --logs ~/scratch/logs \
                    --zeroshot-frequency 1 \
                    --report-to wandb,tensorboard \
                    --wandb_offline \
                    --pathmnist \
                "


#cat srun_worker.sh
#srun bash srun_worker.sh

# --dataset cifar100 --self_batch_size 1024  --learning_rate 0.2 --cosine --syncBN
# --dataset cifar10 --self_batch_size 512 --learning_rate 0.1 --cosine --syncBN
# sbatch -p debug_full_node --mail-type NONE --time '0:30:00' --array 0 run.sh
# sbatch -p debug --gpus-per-node 1 --mail-type NONE --time '2:00:00' --array 0 run.sh
# sbatch -p compute_full_node --gpus-per-node 4 run.sh
