"""Donne la liste des fichiers du corpus."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator


def list_extensions() -> None:
    """Donne la liste des extensions de fichier dans le corpus."""
    corpus: Iterator[Path] = Path("./corpus").rglob("*")
    extensions: set[str] = {file.suffixes[-1] for file in corpus if file.is_file() and file.suffixes}
    for ext in extensions:
        print(ext)


def get_pdf_text(file: Path) -> str:
    """Renvoie le texte d'un pdf."""
    return "\n".join([page.get_text() for page in pymupdf.open(file)])


def main():
    """Renvoie le texte d'un document pdf."""
    # list_extensions()
    # return
    ###
    files: Iterator[Path] = Path("./corpus").rglob("*.jpg")
    file: Path = next(files)

    print("##" * 80)
    print(file)
    print("##" * 80)
    return
    text: str = get_pdf_text(file)
    print(text)
    print("##" * 80)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint: str = "almanach/camembertav2-base"

    with torch.no_grad():
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model: AutoModel = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        model.eval()

        # actual processing of the document chunks
        encoded = tokenizer(
            text,
            # hyper param pour savoir quel overlap de contexte on veut. Un plus grand overlap voudra
            # dire que les segments d'un meme texte seront plus similaires.
            stride=50,
            return_overflowing_tokens=True,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        output = model(input_ids, attn_mask)
        lsh = output.last_hidden_state

        # on utilise unsqueeze pour passer en shape [b, s, 1]
        # puis on expand pour passer en shape [b, s, h].
        # Sans le unsqueeze, on n'aurait pas le bon nombre de dimensions
        # et le expand ne fonctionnerait pas
        am = attn_mask.unsqueeze(-1).expand(lsh.shape)
        embed = (lsh * am).sum(1)  # hadamard product mais c'est ce qu'on veut
        mask = am.sum(1)
        embed = (embed / mask).detach().to(torch.float32).cpu().numpy()
        print(embed.shape)
        print(cosine_similarity(embed))


if __name__ == "__main__":
    main()
