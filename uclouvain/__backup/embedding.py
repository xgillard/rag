from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from preprocessing import preprocess

if TYPE_CHECKING:
    from typing import Callable

    import numpy as np
    from transformers import AutoModel, AutoTokenizer, BatchEncoding


class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray


type EmbeddingPipeline = Callable[[str], list[EmbeddedChunk]]


def embedding_pipeline(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device | str | None = None,
) -> EmbeddingPipeline:
    """Return an easy to (re)use pipeline to compute the embeddings for the various chunks of a document."""
    return functools.partial(
        compute_embeddings,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def compute_embeddings(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device | str | None = None,
) -> list[EmbeddedChunk]:
    """Compute the embedding for the various chunks of a text."""
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text: str = preprocess(text)
    with torch.no_grad():
        model.to(device)
        model.eval()
        # actual processing of the document chunks
        encoded = tokenizer(
            text,
            # hyper param pour savoir quel overlap de contexte on veut. Un plus grand overlap voudra
            # dire que les segments d'un meme texte seront plus similaires.
            stride=20,
            return_overflowing_tokens=True,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        output = model(input_ids, attn_mask)
        lhs = output.last_hidden_state

        # normalisation du lhs
        lhs = lhs / torch.linalg.vector_norm(lhs, dim=-1, keepdim=True)

        # on utilise unsqueeze pour passer en shape [b, s, 1]
        # puis on expand pour passer en shape [b, s, h].
        # Sans le unsqueeze, on n'aurait pas le bon nombre de dimensions
        # et le expand ne fonctionnerait pas
        am: torch.tensor = attn_mask.unsqueeze(-1).expand(lhs.shape)
        ############################################################################
        # average pooling
        # --------------------------------------------------------------------------
        embed = (lhs * am).sum(1)  # hadamard product mais c'est ce qu'on veut
        mask = am.sum(1)
        embed = (embed / mask).detach()
        ############################################################################
        # max pooling
        # --------------------------------------------------------------------------
        # embed: torch.tensor = lhs * am  # hadamard product mais c'est ce qu'on veut
        # embed: torch.tensor = torch.max(embed, dim=1).values
        ############################################################################
        # 3e possibilité: on va plutot utiliser l'embedding du token [CLS]
        # --------------------------------------------------------------------------
        # embed: torch.tensor = lhs[:, 0, :]
        ############################################################################
        # Normalisation
        # --------------------------------------------------------------------------
        # Il faut maintenant normaliser les embeddings pour obtenir des
        # vecteurs de magnitude 1. Pour ca, on divise chaque embedding par
        # sa norme L2.
        ############################################################################
        embed: torch.tensor = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        ############################################################################
        # on finalise les embeddings en amenant le tenseur sur le cpu
        embed: np.ndarray = embed.to(torch.float32).cpu().numpy()
        return [EmbeddedChunk(__original_text(text, encoded, i), embed[i, :]) for i in range(len(embed))]


def __original_text(text: str, encoded: BatchEncoding, i: int) -> str:
    """Return the original text for the ith sample in the batch encoding."""
    offsets: torch.tensor = encoded["offset_mapping"][i]
    start: int = offsets[1, 0]  # (1,0) because (0,0) is the special [CLS] token
    stop: int = offsets.max()
    return text[start : stop + 1]


if __name__ == "__main__":
    import pymupdf
    from transformers import AutoModel, AutoTokenizer

    text: str = "\n".join(page.get_text() for page in pymupdf.open("./example/example.pdf"))

    model_name: str = "intfloat/multilingual-e5-large-instruct"  # "FacebookAI/xlm-roberta-base"
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModel = AutoModel.from_pretrained(model_name)
    manual = compute_embeddings(text, tokenizer=tokenizer, model=model)
    print(manual[-1].embedding.shape)
