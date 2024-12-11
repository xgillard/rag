"""Prefetch the model weights before finishing the image compilation."""

import torch
from transformers import AutoModel, AutoTokenizer

CHECKPT: str = "jinaai/jina-embeddings-v3"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPT)
MODEL: AutoModel = AutoModel.from_pretrained(CHECKPT, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
