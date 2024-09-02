
cd src
export PYTHONPATH="./"

python vqa/run.py \
    experiment_name=vqarad_flip25 \
    dataset=vqarad \
    module.network.model_name=ViT-B-16_vqa \
    module.network.pretrained=/projects/multimodal/checkpoints/methods/wacv_exps/Methods_Flip_ViT_B_16_025/checkpoints/epoch_7.pt \
    job_type=eval \
    resume_from_checkpoint=/checkpoint/yaspar/13487175/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt