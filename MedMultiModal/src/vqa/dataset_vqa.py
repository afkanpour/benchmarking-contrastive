"""
Dictionary module as coded in [1] which itself modifies [2].

Parses all questions in a VQA dataset (e.g. VQA-RAD and PathVQA) and makes a dictionary
of all words that appear in the questions. The dictionary is stored in two maps; one
from word to index, and another from index to word. Moreover, it tokenizes a given
sentence using said maps. Tokens can be padded or truncated to fit a given context
length.

References
----------
[1] Jin-Hwa Kim, "Bilinear Attention Networks", URL: https://github.com/jnhwkim/ban-vqa
[2] Xuan B. Nguyen, "Mixture of Enhanced Visual Features",
    URL: https://github.com/aioz-ai/MICCAI19-MedVQA
"""

from __future__ import annotations, print_function

import pickle
import warnings
from typing import Dict, List, Optional

import torch
from lightning_utilities.core.rank_zero import rank_zero_info


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
COUNTING_ONLY = False


class Dictionary:
    """Keep a map of words to tokens and tokenize any given text.

    Parameters
    ----------
    word2idx: Dict[str, int], optional, default=None
        Map of words to indexes (i.e. tokens).
    idx2word: List[str], optional, default=None
        Map of indexes (i.e. tokens) to words.
    context_length: int, default=12
        Maximum number of tokens per sentence in the `__call__` method.
        Note that this parameter is not used in `tokenize` method; the number of tokens
        returned by `tokenize` is equal to the number of words in the given text.
    """

    def __init__(
        self,
        word2idx: Optional[Dict[str, int]] = None,
        idx2word: Optional[List[str]] = None,
        context_length: int = 12,
    ) -> None:
        """Initialize the module."""
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.context_length = context_length

    @property
    def ntoken(self) -> int:
        """Return number of tokens in the dictionary."""
        return len(self.word2idx)

    @property
    def padding_idx(self) -> int:
        """Return padding token.

        The padding token is one plus the highest token associated with a word.
        """
        return len(self.word2idx)

    def tokenize(self, sentence: str, add_word: bool) -> List[int]:
        """Tokenize given text using the words in the dictionary.

        Parameters
        ----------
        sentence: str
            Text to be tokenized.
        add_word: bool
            How to treat new words that might appear in `sentence` that are not in the
            dictionary. If `add_word` is `True`, new words will be added to the
            dictionary and tokenized accordingly. If `add_word` is `False`, new words
            will be replaced with the last word of the dictionary.
        """
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = (
            sentence.replace(",", "")
            .replace("?", "")
            .replace("'s", " 's")
            .replace("...", "")
            .replace("x ray", "x-ray")
            .replace(".", "")
        )
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced
                # with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path: str) -> None:
        """Dump word-to-index & index-to-word maps to file."""
        with open(path, "wb") as file:
            pickle.dump([self.word2idx, self.idx2word], file)
        rank_zero_info(f"Dictionary dumped to {path}.")

    @classmethod
    def load_from_file(cls, path: str, context_length: int = 12) -> Dictionary:
        """Load word-to-index and index-to-word maps from file."""
        rank_zero_info(f"Loading dictionary from {path}.")
        with open(path, "rb") as file:
            word2idx, idx2word = pickle.load(file)
        return cls(word2idx, idx2word, context_length)

    def add_word(self, word: str) -> int:
        """Add given word to dictionary if not already included.

        Parameters
        ----------
        word: str
            Word to be added.

        Returns
        -------
        int
            Index (i.e. token) assigned to the given word.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        """Return number of tokens in the dictionary."""
        return len(self.idx2word)

    def __call__(self, sentence: str) -> torch.LongTensor:
        """Tokenize, truncate and pad the tokens.

        Tokens will be truncated and/or padded to fit the number given by
        `context_length` attribute.

        Parameters
        ----------
        sentence: str
            Text to be tokenized.
        """
        tokens = self.tokenize(sentence, add_word=False)
        tokens = tokens[: self.context_length]
        if len(tokens) < self.context_length:
            # Note here we pad in front of the sentence
            padding = [self.padding_idx] * (self.context_length - len(tokens))
            tokens = tokens + padding
        assert len(tokens) == self.context_length
        return torch.LongTensor(tokens)
