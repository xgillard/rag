"""Prefetch the model weights before finishing the image compilation."""

import torch
from transformers import AutoModel, AutoTokenizer


def prefetch(checkpoint: str, dtype: torch.dtype) -> None:
    """Prefetch a model so that it is stored in the image at build time."""
    _tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    _model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype, trust_remote_code=True)


cuda = torch.cuda.is_available()

prefetch("jinaai/jina-embeddings-v3", torch.bfloat16 if cuda else torch.float32)
prefetch("intfloat/multilingual-e5-large-instruct", torch.float16)
prefetch("intfloat/multilingual-e5-small", torch.float32)
