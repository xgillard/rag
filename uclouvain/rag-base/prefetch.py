"""Prefetch the model weights before finishing the image compilation."""

#import torch
#from transformers import AutoModel, AutoTokenizer
#
#
#def prefetch(checkpoint: str, dtype: torch.dtype) -> None:
#    """Prefetch a model so that it is stored in the image at build time."""
#    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#    model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype, trust_remote_code=True)
#    # save the models locally so that they can later be mounted easily.
#    tokenizer.save_pretrained(f"./models/{checkpoint}")
#    model.save_pretrained(f"./models/{checkpoint}")
#
#
#cuda = torch.cuda.is_available()
#prec = torch.bfloat16 if cuda else torch.float32
#
#prefetch("jinaai/jina-embeddings-v3", prec)
#prefetch("intfloat/multilingual-e5-large-instruct", torch.float16)
#prefetch("intfloat/multilingual-e5-small", prec)
#prefetch("meta-llama/Llama-3.2-1B-Instruct", prec)

#from sentence_transformers import SentenceTransformer
#
#prefetch = SentenceTransformer("intfloat/multilingual-e5-small")
#prefetch.save_pretrained("./models/intfloat/multilingual-e5-small")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("./models/meta-llama/Llama-3.2-3B-Instruct")
tokenizer.save_pretrained("./models/meta-llama/Llama-3.2-3B-Instruct")