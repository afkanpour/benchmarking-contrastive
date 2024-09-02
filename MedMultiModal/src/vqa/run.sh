
cd src
export PYTHONPATH="./"

python vqa/run.py \
    experiment_name=pathvqa_flip50 \
    module.network.model_name=ViT-B-16_vqa \
    module.network.pretrained=/projects/multimodal/checkpoints/methods/Methods_Flip_ViT_B_16_05/checkpoints/epoch_7.pt \
    job_type=train \
    resume_from_checkpoint=/checkpoint/fogidi/12736266/last.ckpt