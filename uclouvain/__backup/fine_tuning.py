from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf

from preprocessing import preprocess


def get_text(fname: Path) -> str:
    """Extrait le texte d'un document pdf (sans utiliser l'ocr)"""
    return "\n".join([page.get_text() for page in pymupdf.open(fname)])


def prepare_dataset(folder: Path) -> pd.DataFrame:
    """Recursively index all files from the given folder.

    The text is first extracted from each document, then converted
    to embeddings using the specified model. After that, the embeddings
    are stored in the vector database.
    """
    data = []
    for file in folder.rglob("*.pdf"):
        try:
            text: str = preprocess(get_text(file))
            data.append({"fname": f"{file}", "text": text})
        except:
            print(f"problem while processing {file}")
    return pd.DataFrame(data=data)


if __name__ == "__main__":
    df: pd.DataFrame = prepare_dataset(Path("./corpus/"))
    df.to_csv("corpus_pdf.csv", header=True, index=False)
