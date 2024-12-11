"""Provides functionality to compute embedding using the selected model."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .preprocessing import preprocess

if TYPE_CHECKING:
    import numpy as np

###################################################################################
CHECKPT: str = "jinaai/jina-embeddings-v3"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPT)
MODEL: AutoModel = AutoModel.from_pretrained(CHECKPT, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
# hyper param
STRIDE: int = 20
MAX_LENGTH: int = 8192
# lower runtime impact of the model
MODEL.eval()
###################################################################################


class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray


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
    # this is specific to "jinaai/jina-embeddings-v3"
    task_id = MODEL._adaptation_map[str(task)]  # noqa: SLF001
    adapter_mask = torch.full((input_ids.shape[0],), task_id, dtype=torch.int32)
    # compute the output
    output = MODEL(input_ids, attn_mask, adapter_mask=adapter_mask)

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
