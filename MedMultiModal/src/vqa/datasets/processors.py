"""All preprocessors for textual inputs."""

from typing import Callable, List, Optional, Union

import clip
import torch


class VQAProcessor(object):
    """Preprocessor for textual reports of VQA datasets."""

    def __call__(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
        """Process the textual captions."""
        if not isinstance(sentence, (list, str)):
            raise TypeError(
                f"Expected sentence to be a string or list of strings, got {type(sentence)}"
            )

        def _preprocess_sentence(sentence: str) -> str:
            sentence = sentence.lower()
            if "? -yes/no" in sentence:
                sentence = sentence.replace("? -yes/no", "")
            if "? -open" in sentence:
                sentence = sentence.replace("? -open", "")
            if "? - open" in sentence:
                sentence = sentence.replace("? - open", "")
            return (
                sentence.replace(",", "")
                .replace("?", "")
                .replace("'s", " 's")
                .replace("...", "")
                .replace("x ray", "x-ray")
                .replace(".", "")
            )

        if isinstance(sentence, str):
            return _preprocess_sentence(sentence)

        for i, s in enumerate(sentence):
            sentence[i] = _preprocess_sentence(s)

        return sentence


class TrimText(object):
    """Trim text strings as a preprocessing step before tokenization."""

    def __init__(self, trim_size: int) -> None:
        """Initialize the object."""
        self.trim_size = trim_size

    def __call__(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
        """Trim the given sentence(s)."""
        if not isinstance(sentence, (list, str)):
            raise TypeError(
                "Expected argument `sentence` to be a string or list of strings, "
                f"but got {type(sentence)}"
            )

        if isinstance(sentence, str):
            return sentence[: self.trim_size]

        for i, s in enumerate(sentence):
            sentence[i] = s[: self.trim_size]

        return sentence


class ClipTokenizer(object):
    """OpenAI's CLIP model tokenizer plus custom preprocessing.

    Preprocessing is as required for MedVQA datasets such as PathVQA and VQARAD.
    """

    def __init__(
        self,
        context_length: int = 77,
        truncate: bool = False,
        processor: Optional[
            Callable[[Union[str, List[str]]], Union[str, List[str]]]
        ] = None,
    ) -> None:
        """Initialize the object with the preprocessing function.

        Parameters
        ----------
        context_length : int, default=77
            Number of tokens per given sentence.
        truncate : bool, default=False
            Whether to truncate the tokenized sentence if longer than the context
            length.
        processor : Callable[[str | List[str]], str | List[str]], optional, default=None
            Function to preprocess the text string before tokenization.
        """
        self.context_length = context_length
        self.truncate = truncate
        self.processor = processor

    def __call__(self, sentence: Union[str, List[str]]) -> torch.Tensor:
        """Tokenize a text using OpenAI's CLIP model.

        Parameters
        ----------
        sentence : str or list of str
            Sentence(s) to be tokenized.

        Returns
        -------
        torch.Tensor
            Tokenized sentence(s).
        """
        if self.processor is not None:
            sentence = self.processor(sentence)
        return clip.tokenize(sentence, self.context_length, self.truncate).squeeze()
