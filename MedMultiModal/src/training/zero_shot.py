import logging

import torch
from tqdm import tqdm

from open_clip import (
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
    build_zero_shot_classifier,
    get_input_dtype,
    get_tokenizer,
)
from open_clip.zero_shot_metadata import PATHMNIST_CLASSNAMES, PATHMNIST_TEMPLATES

from .precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = (
                    output["image_features"] if isinstance(output, dict) else output[0]
                )
                logits = 100.0 * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if (
        "imagenet-val" not in data
        and "imagenet-v2" not in data
        and "pathmnist" not in data
    ):
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info("Starting zero-shot imagenet.")
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info("Building zero-shot classifier")
    autocast = get_autocast(args.precision)
    with autocast():
        if "pathmnist" in data:
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=PATHMNIST_CLASSNAMES,
                templates=PATHMNIST_TEMPLATES,
                num_classes_per_batch=1,
                device=args.device,
                use_tqdm=True,
            )
        else:
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=1,
                device=args.device,
                use_tqdm=True,
            )

    logging.info("Using classifier")
    results = {}
    if "imagenet-val" in data:
        top1, top5 = run(model, classifier, data["imagenet-val"].dataloader, args)
        results["imagenet-zeroshot-val-top1"] = top1
        results["imagenet-zeroshot-val-top5"] = top5
    if "imagenet-v2" in data:
        top1, top5 = run(model, classifier, data["imagenet-v2"].dataloader, args)
        results["imagenetv2-zeroshot-val-top1"] = top1
        results["imagenetv2-zeroshot-val-top5"] = top5
    if "pathmnist" in data:
        top1, top5 = run(model, classifier, data["pathmnist"].dataloader, args)
        results["pathmnist-zeroshot-val-top1"] = top1
        results["pathmnist-zeroshot-val-top5"] = top5

    logging.info("Finished zero-shot imagenet.")

    return results
