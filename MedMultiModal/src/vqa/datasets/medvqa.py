"""Module to load the following datasets: PathVQA and VQARAD."""

import json
import os
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Grayscale, Resize, ToTensor
from vqa.datasets.processors import ClipTokenizer, VQAProcessor

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


class MedVQA(Dataset):
    """Module to load PathVQA and VQARAD datasets.

    Parameters
    ----------
    root_dir: str
        Path to the root directory of the dataset.
    split : {"train", "val", "test"}, default = "train"
        Split of the dataset to use.
    encoder: dict
        image_size: int
            Size of the input images; e.g. 224 for clipvision
    autoencoder: dict
        available: boolean {True, False}
            Whether or not to return autoencoder images.
        image_size: int
            Size of the input images; e.g. 128
    num_ans_candidates: int
        Number of all unique answers in the dataset.
    rgb_transform: Optional[Callable[[Image.Image], torch.Tensor]]
        Transform applied to images that will be passed to the visual encoder.
    ae_transform: Optional[Callable[[Image.Image], torch.Tensor]]
        Transform applied to images that will be passed to the autoencoder.
    tokenizer : Optional[Callable[[str | List[str]], torch.Tesnor]]
        Function to tokenize the questions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val", "test"],
        encoder: Dict[str, Any],
        autoencoder: Dict[str, Any],
        num_ans_candidates: int,
        rgb_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        ae_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[Callable[[Union[str, List[str]]], torch.Tensor]] = None,
    ) -> None:
        """Initialize the dataset."""
        super(MedVQA, self).__init__()
        assert split in ["train", "val", "test"]
        self.autoencoder = autoencoder["available"]
        self.num_ans_candidates = num_ans_candidates

        # transform for encoder images
        if rgb_transform is None:
            self.rgb_transform = Compose(
                [
                    Resize(encoder["image_size"]),
                    CenterCrop(encoder["image_size"]),
                    ToTensor(),
                ]
            )
        else:
            self.rgb_transform = rgb_transform

        # transform for autoencoder images
        if self.autoencoder and ae_transform is None:
            self.ae_transform = Compose(
                [
                    Grayscale(1),
                    Resize(autoencoder["image_size"]),
                    CenterCrop(autoencoder["image_size"]),
                    ToTensor(),
                ]
            )
        elif self.autoencoder:
            self.ae_transform = ae_transform

        # tokenizer for textual questions
        if tokenizer is None:
            self.tokenize_fn: Callable[[Union[str, List[str]]], torch.Tensor] = (
                ClipTokenizer(context_length=77, processor=VQAProcessor())
            )
        else:
            self.tokenize_fn = tokenizer

        # load entries
        with open(
            os.path.join(root_dir, "cache", f"{split}_data.json"), encoding="utf-8"
        ) as file:
            self.entries = json.load(file)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an example/sample of the data.

        Returns
        -------
        example: Dict[str, Any]
            One sample of the dataset.
            "text": torch.Tensor
                Question as tokens.
            "rgb": torch.Tensor | List[torch.Tensor]
                Preprocessed image.
                A list of two torch Tensors if `autoencoder.available` is set
                True in the dataset config, otherwise a single torch Tensor.
            "rgb_target": torch.Tensor
                Multi-hot-encoding of the correct answer classes as a vector.
            "qid": int
                The qid of the sample.
            "answer_type": str {"yes/no", "number", "OPEN", "CLOSED", ...}
                Answer type.
            "question_type": str {"what", "does", "are", "SIZE", "PRES", ...}
                Question type.
            "phrase_type": str {"freeform", "frame"} | int {-1}
                Phrase type.
                (-1 in case the dataset does not have phrase_type info).
            "raw_question": str
                Question as text.

        Notes
        -----
        If `autoencoder.available` is set True in the dataset configs, a list
        of two torch Tensors are returned as `"rgb"`. The first element
        of the list is the processed image meant for the visual encoder and
        the second element is the image meant for the autoencoder in the MEVF
        pipeline (see [1] for more information). If `autoencoder.available` is
        False, only the image meant for the encoder is returned.

        References
        ----------
        [1] Nguyen, Binh D., Thanh-Toan Do, Binh X. Nguyen, Tuong Do, Erman
        Tjiputra, and Quang D. Tran. "Overcoming data limitation in medical
        visual question answering." In Medical Image Computing and Computer
        Assisted Intervention, MICCAI 2019: 22nd International Conference.
        """
        entry = self.entries[index]
        question = self.tokenize_fn(entry["question"])
        answer = entry["answer"]

        # prepare encoder image
        images_data = Image.open(entry["image_path"]).convert("RGB")
        enc_images_data = self.rgb_transform(images_data)

        # prepare autoencoder image
        if self.autoencoder:
            ae_images_data = self.ae_transform(images_data)

        # pack images if needed
        if self.autoencoder:
            image_data = [enc_images_data, ae_images_data]
        else:
            image_data = enc_images_data

        example = {
            "text": question,
            "rgb": image_data,
            "qid": entry["qid"],
            "answer_type": entry["answer_type"],
            "question_type": entry["question_type"],
            "phrase_type": entry["phrase_type"],
            "raw_question": entry["question"],
        }

        if answer is not None:
            labels = answer["labels"]
            scores = answer["scores"]
            target = torch.zeros(self.num_ans_candidates)
            if len(labels):
                labels = torch.from_numpy(np.array(answer["labels"]))
                scores = torch.from_numpy(np.array(answer["scores"], dtype=np.float32))
                target.scatter_(0, labels, scores)
            example["rgb_target"] = target

        return example

    def __len__(self) -> int:
        """Return size of the dataset."""
        return len(self.entries)
