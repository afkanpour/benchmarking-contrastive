"""
Utility functions for VQA pipeline as coded in [1].

References
----------
[1] Jin-Hwa Kim, "Bilinear Attention Networks", URL: https://github.com/jnhwkim/ban-vqa
"""

from __future__ import print_function

import itertools
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from omegaconf import DictConfig
from vqa.dataset_vqa import Dictionary
from vqa.language_model import WordEmbedding


def tfidf_loading(
    use_tfidf: bool, w_emb: WordEmbedding, cfg: DictConfig
) -> WordEmbedding:
    """Load TF-IDF weights in `WordEmedding` module.

    Parameters
    ----------
    use_tfidf: bool
        Whether or not to load TF-IDF weights in `WordEmedding` module.
    w_emb: WordEmedding
        Word embedding module in which TF-IDF weights are loaded.
    cfg: DictConfig
        Configuration of the whole MEVF pipeline.
    """
    if use_tfidf:
        # load extracted tfidf and weights from file for saving loading time
        if os.path.isfile(
            os.path.join(cfg.question_embedding.dict_root, "embed_tfidf_weights.pth")
        ):
            rank_zero_info("Loading embedding tfidf and weights from file.")
            w_emb.load_state_dict(
                torch.load(
                    os.path.join(
                        cfg.question_embedding.dict_root, "embed_tfidf_weights.pth"
                    )
                )
            )
            rank_zero_info("Loaded embedding tfidf and weights from file successfully.")
        else:
            rank_zero_info(
                "Embedding tfidf and weights haven't been saved before.\nComputing tfidf and weights from start."
            )
            dictionary = Dictionary.load_from_file(
                os.path.join(cfg.question_embedding.dict_root, "dictionary.pkl")
            )
            tfidf, weights = tfidf_from_questions(["train", "test"], cfg, dictionary)
            w_emb.init_embedding(
                os.path.join(cfg.question_embedding.dict_root, "glove6b_init_300d.npy"),
                tfidf,
                weights,
            )
            if rank_zero_only.rank == 0:
                torch.save(
                    w_emb.state_dict(),
                    os.path.join(
                        cfg.question_embedding.dict_root, "embed_tfidf_weights.pth"
                    ),
                )
                print("Saved embedding with tfidf and weights successfully.")
    return w_emb


def tfidf_from_questions(
    names: List[str],
    cfg: DictConfig,
    dictionary: Dictionary,
    dataroot: str = "data",
    target: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute TF-IDF stochastic matrix from questions in a VQA dataset.

    This method is tested for VQARAD and PathVQA datasets.
    """
    if target is None:
        target = ["rad"]
    dataroot = cfg.question_embedding.dict_root
    rank_zero_warn(
        f"Please ensure that dictionary root path ({dataroot}) matches the dataset root."
    )

    inds: List[List[int]] = [[], []]  # rows, cols for uncoalesce sparse matrix
    df: Dict[int, int] = {}
    n: int = len(dictionary)

    def populate(inds: List[List[int]], df: Dict[int, int], text: str) -> None:
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < n:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < n:
                inds[0].append(c[1])
                inds[1].append(c[0])

    if "rad" in target:
        for name in names:
            assert name in ["train", "val", "test"]
            question_path = os.path.join(dataroot, name + "set.json")
            if not os.path.exists(question_path):
                raise RuntimeError(f"JSON file does not exist: {question_path}.")
            with open(question_path, encoding="utf-8") as file:
                questions = json.load(file)
            for question in questions:
                populate(inds, df, question["question"])

    # TF-IDF
    vals: List[float] = [1] * len(inds[1])
    for _, col in enumerate(inds[1]):
        assert df[col] >= 1, "document frequency should be greater than zero!"
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds: List[List[int]], vals: List[float]) -> List[float]:
        z: Dict[int, float] = {}
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, "glove", f"glove.6B.{emb_dim}d.txt")
    weights, word2emb = create_glove_embedding_init(dictionary.idx2word[n:], glove_file)
    rank_zero_info(
        f"TF-IDF stochastic matrix ({tfidf.size(0)} x {tfidf.size(1)}) is generated."
    )

    return tfidf, weights


def create_glove_embedding_init(
    idx2word: List[str],
    glove_file: str,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create GloVe embedding file."""
    word2emb = {}

    with open(glove_file, "r", encoding="utf-8") as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(" ")) - 1
    rank_zero_info(f"embedding dim is {emb_dim}")
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(" ")
        word = vals[0]
        word2emb[word] = np.array([float(x) for x in vals[1:]])
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def get_ntokens_from_dict(dict_root_dir: str) -> int:
    """Load a dictionary from file and return its number of tekens."""
    with open(os.path.join(dict_root_dir, "dictionary.pkl"), "rb") as file:
        word2idx, _ = pickle.load(file)
    return len(word2idx)
