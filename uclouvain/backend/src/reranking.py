"""Provides functionality to compute reranking embedding using the selected model.

The code from this file is almost duplicated from the 'retrieval.py' file. However,
it does not use the LoRA adapters that are used to tune the retrieval. This one
uses a prompt instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .embedding import EmbeddedChunk
from .preprocessing import preprocess

if TYPE_CHECKING:
    import numpy as np

###################################################################################
CHECKPT: str = "intfloat/multilingual-e5-large-instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPT)
DTYPE: torch.dtype = torch.float16
MODEL: AutoModel = AutoModel.from_pretrained(CHECKPT, torch_dtype=DTYPE).to(DEVICE)
# hyper param
STRIDE: int = 20
MAX_LENGTH: int = 512
# lower runtime impact of the model
MODEL.eval()
###################################################################################


def encode_document(text: str) -> list[EmbeddedChunk]:
    """Compute the embeddings for the various chunks of a document.

    The document is 1st chunked by the tokenizer and then document
    embedding is computed for each chunk.
    """
    return compute_embeddings(text, "retrieval.passage")


def encode_query(text: str) -> np.ndarray:
    """Compute one single embedding meant to represent the whole user query.

    In the event where the user query is very long, the tokenizer will split
    it in several contexts and the model will produce one embedding for each
    chunk. These many embeddings, however, are not really useful when it comes
    to retrieving relevant documents to answer the query. This is why, in the
    event of a very long query, the chunks embeddings are pooled together so
    as to produce one single vector to encode the user request.
    """
    ### INSTRUCT MODEL ######################################################
    prompt: str = """Given a user request, attempt to find the most relevant
    piece of information in order to satisfy that user's request.

    # User Request:
    """
    text: str = prompt + text
    #########################################################################
    chunks: np.ndarray = np.stack([e.embedding for e in compute_embeddings(text, "retrieval.query")])
    return np.mean(chunks, axis=0)


@torch.no_grad()
def compute_embeddings(
    text: str,
    task: str,
) -> list[EmbeddedChunk]:
    """Compute the embedding for the various chunks of a text.

    *Important note:*
    The `task` param must be either "retrieval.passage" or ""retrieval.query"
    """
    text: str = preprocess(text)

    # actual processing of the document chunks
    encoded = TOKENIZER(
        text,
        stride=STRIDE,
        return_overflowing_tokens=True,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attn_mask = encoded["attention_mask"].to(DEVICE)
    ##################################################################################
    ## this is specific to "jinaai/jina-embeddings-v3"
    # task_id = MODEL._adaptation_map[str(task)]
    # adapter_mask = torch.full((input_ids.shape[0],), task_id, dtype=torch.int32)
    # output = MODEL(input_ids, attn_mask, adapter_mask=adapter_mask)
    ##################################################################################
    ## compute the output
    output = MODEL(input_ids, attn_mask)

    lhs: torch.tensor = output.last_hidden_state  # shape: [b, s, h]
    embed: torch.tensor = __avg_pooling(lhs, attn_mask)  # shape: [b, h]
    embed: torch.tensor = __normalize(embed)  # shape: [b, h]
    embed: np.ndarray = embed.to(torch.float32).cpu().numpy()
    return [EmbeddedChunk(__original_text(text, encoded, i), embed[i, :]) for i in range(len(embed))]


def __avg_pooling(last_hidden_state: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    """Perform a pass of average pooling on the given input tensor."""
    # unsqueeze to get a tensor of shape [b, s, 1]
    # expand bcast to  a tensor of shape [b, s, h].
    am: torch.tensor = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
    embed = (last_hidden_state * am).sum(1)
    mask = am.sum(1)
    return (embed / mask).detach()


def __normalize(embedding: torch.tensor) -> torch.tensor:
    """Normalize the magnitude of the embedding tensor (L2 normalization)."""
    return embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)


def __original_text(text: str, encoded: BatchEncoding, i: int) -> str:
    """Return the original text for the ith sample in the batch encoding."""
    offsets: torch.tensor = encoded["offset_mapping"][i]
    start: int = offsets[1, 0]  # (1,0) because (0,0) is the special [CLS] token
    stop: int = offsets.max()
    return text[start : stop + 1]
