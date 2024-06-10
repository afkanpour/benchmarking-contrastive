# learning rate = 1e-5
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-0/checkpoints/epoch_3.pt
# learning rate = 2e-5
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-1/checkpoints/epoch_4.pt
# learning rate = 5e-5
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-2/checkpoints/epoch_7.pt
# learning rate = 1e-4
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-3/checkpoints/epoch_7.pt
# learning rate = 2e-4
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-4/checkpoints/epoch_8.pt
# learning rate = 5e-4
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-5/checkpoints/epoch_14.pt


# batch size = 8
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_8/checkpoints/epoch_6.pt
# batch size = 16
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_16/checkpoints/epoch_8.pt
# batch size = 32
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_32/checkpoints/epoch_6.pt
# batch size = 64
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_64/checkpoints/epoch_7.pt
# batch size = 128
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_128/checkpoints/epoch_7.pt
# batch size = 256
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_256/checkpoints/epoch_6.pt
# batch size = 512
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_512/checkpoints/epoch_8.pt
# batch size = 1024
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/ViT-B-32-ALL-2/checkpoints/epoch_7.pt

# GPT/77 & ViT-B/16
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/encoder_combination/gpt77-ViT-B-16-ALL/checkpoints/epoch_8.pt
# GPT/77 & ViT-B/32
MODEL=ViT-B-32
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/hp_tuning/HPtuning_batchsize_32/checkpoints/epoch_6.pt
# GPT/77 & ViT-L/14
MODEL=ViT-L-14
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/encoder_combination/gpt77-ViT-L-14-ALL/checkpoints/epoch_3.pt
# GPT/77 & RN50
MODEL=RN50
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/encoder_combination/gpt77-RN-50-ALL/checkpoints/epoch_11.pt

# LockText
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_LockText/checkpoints/epoch_7.pt
# LockImage
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_LockImage_ViT_B_16/checkpoints/epoch_17.pt
# Flip - ratio=0.75
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Flip_ViT_B_16/checkpoints/epoch_7.pt
# Unimodal Image
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_Unimodal_Image_ViT_B_16/checkpoints/epoch_16.pt
# Unimodal Masked Image
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_Unimodal_Masked_Image_ViT_B_16/checkpoints/epoch_16.pt
# TextUnloced 2 (83%)
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_TextUnlocked_2_ViT_B_16/checkpoints/epoch_3.pt
# TextUnloced 6 (50%)
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_TextUnlocked_6_ViT_B_16/checkpoints/epoch_6.pt
# ImageUnloced 7 (50%)
MODEL=ViT-B-16
PRETRAINED_WEIGHTS=/projects/multimodal/checkpoints/methods/Methods_Clip_ImageUnlocked_7_ViT_B_16/checkpoints/epoch_6.pt
# CLIP+CoCa
MODEL=coca_ViT-B-32
PRETRAINED_WEIGHTS="/projects/multimodal/checkpoints/methods/Methods_ClipCoca_ViT_B_16/checkpoints/epoch_5.pt --caption-encoder ViT-B-16"
# FILIP
MODEL=ViT-B-16-FILIP
PRETRAINED_WEIGHTS="openai --filip --filip_checkpoint /checkpoint/yaspar/12707961/Methods_FIlip_ViT_B_16/checkpoints/epoch_1.pt"

