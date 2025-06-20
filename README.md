## Getting Started
1. Clone this repository:
```bash
git clone <link to this repository>
```

2. Create and activate a virtual environment:
```bash
python3 -m venv /path/to/new/virtual/environment/multimodal_env
source /path/to/new/virtual/environment/multimodal_env/bin/activate
```

3. Install the requirements:
```bash
cd MedMultiModal
pip install -r requirements.txt
```

This codebase is tested with python3.7 and torch 1.13 built with cuda 11.7

_Note:_ we have extended the code of two well-tested repositories to run our experiments: open_clip [[1]](#1) and clip_benchmark [[2]](#2).

<!-- ## Datasets
### Pre-training

### Evaluation -->

## Usage
### Pre-training
1. Move to the `src` directory:
```bash
cd src
```

2. Add the directory to `PYTHONPATH`:
```bash
export PYTHONPATH="./"
```

3. Run the training script. For example, the following command trains a model with ViT-B/16 visual encoder and GPT/77 text encoder, on a combination of four medical datasets, and logs the results using Weights & Biases.
The model is loaded with pretrained weights from "openai", i.e. pretrained on the ImageNet dataset.
```bash
python -u training/main.py \
    --model ViT-B-16 \
    --pretrained openai \
    --train-data pmc_oa/train.jsonl::Quilt_1M/quilt_1m_train.csv::/mimic_cxr/mimic_cxr_double_image_train.csv::roco/cache/radiologytraindata.csv \
    --train-num-samples 2769337 \
    --dataset-type mixed \
    --csv-separator , \
    --val-data pmc_oa/valid.jsonl \
    --val-no-retrieval \
    --batch-size 128 \
    --accum-freq 4 \
    --workers 4 \
    --lr 5e-5 \
    --lr-scheduler cosine \
    --epochs 20 \
    --warmup 0 \
    --aug-cfg quilt_crop=True \
    --wd 0.1 \
    --name GPT-77-ViT-B-16\
    --resume latest \
    --gather-with-grad \
    --logs /checkpoint/$USER/$SLURM_JOBID/ \
    --zeroshot-frequency 1 \
    --report-to wandb
```

### Downstream Evaluation
1. Move to the `src` directory:
```bash
cd src
```

2. Add the directory to `PYTHONPATH`:
```bash
export PYTHONPATH="./"
```

3. Run the evaluation script. 
For example, the following command runs retrieval on the ROCO dataset.
```bash
python -u clip_benchmark/cli.py eval \
    --dataset roco \
    --task zeroshot_retrieval \
    --model ViT-B-16 \
    --pretrained openai \
    --output result_roco.json \
    --recall_k 1 50 200 \
    --batch_size 64
```

Example of zero-shot classification:
```bash
python clip_benchmark/cli.py eval \
    --dataset pad_ufes_20 \
    --split test \
    --dataset_root datasets/pad_ufes_20 \
    --task zeroshot_classification  \
    --model ViT-B-16 \
    --pretrained openai \
    --output result_pad_ufes_20.json
```

Example of linear probe:
```bash
python -u clip_benchmark/cli.py eval \
    --dataset pad_ufes_20 \
    --train_split train \
    --split test \
    --dataset_root datasets/pad_ufes_20 \
    --task linear_probe \
    --model ViT-B-16 \
    --pretrained openai \
    --output result_pad_ufes_20.json \
    --fewshot_lr 0.1 \
    --fewshot_epochs 40 \
    --batch_size 64
```


Example of VQA:
```bash
python vqa/run.py experiment_name=pathvqa_test module.network.pretrained=<path_to_pretrained_model>
```

## References
<a id="1">[1]</a> open_clip: https://github.com/mlfoundations/open_clip

<a id="2">[2]</a> clip_benchmark: https://github.com/LAION-AI/CLIP_benchmark
