# multimodal-neurips2024

# Pre-training 
```
torchrun -m training.main \
        --model ViT-B-32 \
        --pretrained openai \
        --train-data ./data/quilt/quilt_1M_lookup.csv \
        --train-num-samples 1017708 \
        --dataset-type mixed \
        --csv-separator , \
        --batch-size 256 \
        --accum-freq 4 \
        --workers 8 \
        --lr 1e-5 \
        --lr-scheduler const \
        --epochs 15 \
        --warmup 200 \
        --aug-cfg quilt_crop=True \
        --wd 0.1 \
        --name ViT-B-32-QUILT \
        --resume latest \
        --gather-with-grad \
        --logs ~/scratch/logs \
        --zeroshot-frequency 1 \
        --report-to wandb,tensorboard \
        --wandb_offline \
        --pathmnist
```
## Pre-tranining methods
### Method 1. CLIP
Default setup 

### Method 2. CLIP + CoCa
- Change model with `model coca_ViT-B-32`
To use ViT-B/16 encoder with openai weight:

```
--model coca_ViT-B-32 --pretrained laion2b_s13b_b90k --caption-encoder ViT-B-16
```

### Method 3. Train only one modality 
- Use  `--lock-text` to freeze text branch
- Use `--lock-image` to freeze image branch

### Method 4. Train partial encoder 
- First lock the enocder 
- The use `--lock-image-unlocked-groups 2` to train last 2 layers
- Use `--lock-text-unlocked-layers` for text

### Method 5. FLIP
- Add `--force_patch_dropout 0.75` to set mask ration of 75%

### Method 6. Unimodal Contrastive leanrnig on Image + CLIP
- Add `--image-contrast`
- Add `--aug-cfg num_views=2`


### Method 7. Unimodal Masked Contrastive leanrnig on Image + CLIP
- Add `--image-contrast`
- Add `--aug-cfg num_views=2`
- Add `--mask-contrast`



## Other Training options
### Training on PMC-OA
```bash
--train-data ./data/pmc_oa/train.jsonl --train-num-samples 1317219 --dataset-type json
```

### Training on Quilt-1M
```bash
--train-data ./data/quilt/quilt_1M_lookup.csv --train-num-samples 1017708 --dataset-type mixed --csv-separator , \
```

### Training on MIMIC-CXR
```bash
--train-data ./data/mimic_cxr/mimic_cxr_single_image_train.csv --train-num-samples 222758 --dataset-type csv --csv-separator , \
--csv-img-root path/to/where/images/are/stored --csv-img-key image --csv-caption-key caption
```

### Training Quilt and PMC-QA combined
```bash
--train-data ./data/pmc_oa/train.jsonl::./data/quilt/quilt_1M_lookup.csv --train-num-samples 2334927 --dataset-type mixed --csv-separator , \
```

## Training from different pre-trained encoder

### loading BiomedCLIP
```bash
--model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
```


### Loading Quilt
```bash
--model hf-hub:wisdomik/QuiltNet-B-16 
or 
--model hf-hub:wisdomik/QuiltNet-B-32 
 ```

### Loading our own trained encoder
```bash
--model ViT-B-32 \
--pretrained ./logs/checkpoint.pt
 ```


# Evaluation

## Zero-Shot

### Testing on all datasts currently supported
```bash
python clip_benchmark/cli.py eval \
        --dataset pcam nck lc25000_lung lc25000_colon bach sicap  \
        --task=zeroshot_classification  \
        --model=ViT-L-14 \
        --pretrained=datacomp_xl_s13b_b90k
```


### Testing PCAM on a custom checkpoint
```bash
--model=coca_ViT-B-32 \
--pretrained=./logs/ViT-B-16\/checkpoints/epoch_45.pt
```

## Steps for adding new dataset for ZSL
- add class in src/clip_benchmark/dataset/medical_datasets.py
- add a condifion for load this class in src/clip_benchmark/dataset/builder.py
- add classnames in src/clip_benchmark/dataset/en_classname.json
- add prompt templates in src/clip_benchmark/dataset/en_zeroshot_classification_templates.json

## Currently supported dataset
- pcam
- nck
- bach
- lc25000_colon
- lc25000_lung
- sicap

## Linear probing example from CLIP_benchmark
```bash
python clip_benchmark/cli.py eval --dataset=cifar10 --task=linear_probe --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64 --fewshot_lr 0.1 --fewshot_epochs 20 --batch_size 512 --train_split train --test_split test
```

## Retrieval
```bash 
python clip_benchmark/cli.py eval --dataset=roco --task=zeroshot_retrieval --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64
```

