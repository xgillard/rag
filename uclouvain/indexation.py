"""The purpose of this script is to provide the document indexing facilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2
import pymupdf

from embedding import EmbeddedChunk, embedding_pipeline

if TYPE_CHECKING:
    from psycopg2.extensions import connection

    from .embedding import EmbeddingPipeline


def get_database() -> connection:
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="rag_ucl",
        user="postgres",
        password="admin",
    )


def get_text(fname: Path) -> str:
    """Extrait le texte d'un document."""
    return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(fname)])


def index(
    folder: Path,
    pipeline: EmbeddingPipeline,
    conn: connection,
):
    """Recursively index all files from the given folder.

    The text is first extracted from each document, then converted
    to embeddings using the specified model. After that, the embeddings
    are stored in the vector database.
    """
    with conn.cursor() as cur:
        for file in folder.rglob("*.pdf"):
            text: str = get_text(file)
            chunks: list[EmbeddedChunk] = pipeline(text)

            for chunk in chunks:
                cur.execute(
                    "INSERT INTO documents(path_to_doc, text, embedding) VALUES(%s, %s, %s);",
                    (f"{file}", chunk.text, chunk.embedding.tolist()),
                )
            print(f"DONE : {file}")
        conn.commit()


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    model_name = "almanach/camembertav2-base"
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModel = AutoModel.from_pretrained(model_name)
    pipeline: EmbeddingPipeline = embedding_pipeline(tokenizer=tokenizer, model=model, device="cuda")

    conn: connection = get_database()
    index(Path("./exemple"), pipeline, conn)
