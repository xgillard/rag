"""Prefetch the model weights before finishing the image compilation."""

import torch
from transformers import AutoModel, AutoTokenizer

CHECKPT: str = "intfloat/multilingual-e5-large-instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16
TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPT)
MODEL: AutoModel = AutoModel.from_pretrained(CHECKPT, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
