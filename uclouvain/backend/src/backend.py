"""Main entry point of the RAG application backend.

It exposes all of the REST api that must be made available to an application
developer willing to conduct semantic search & indexation.
"""

from __future__ import annotations

import os

import psycopg2
import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile

from .embedding import EmbeddedChunk, compute_embeddings

app = FastAPI()


@app.post("/indexation/")
async def index_file(path_to_doc: str, file: UploadFile) -> str | None:
    """Perform the indexation of one single file sent over http."""
    conn = get_database()
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(text) FROM documents WHERE path_to_doc = %s;", (f"{path_to_doc}",))
            cnt: int = cur.fetchone()[0]
            if cnt:
                return f"SKIP {path_to_doc} -- {cnt}"
            text: str = get_text(file)
            chunks: list[EmbeddedChunk] = compute_embeddings(text, "retrieval.passage")

            for chunk in chunks:
                cur.execute(
                    "INSERT INTO documents(path_to_doc, text, embedding) VALUES(%s, %s, %s);",
                    (f"{path_to_doc}", chunk.text, chunk.embedding.tolist()),
                )
            conn.commit()
            return f"DONE : {path_to_doc}"  # noqa: TRY300
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"FAIL : {path_to_doc} -- {e}")  # noqa: B904
        return None


def get_text(file: UploadFile) -> str:
    """Extract text from the given file without using ocr."""
    return "\n".join([page.get_text() for page in pymupdf.open(file.file)])


def get_ocr_text(file: UploadFile) -> str:
    """Extract text from the given file and use ocr as needed."""
    return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(file.file)])


def get_database() -> psycopg2.connection:
    """Open a connection to the database."""
    return psycopg2.connect(
        host=os.environ["DATABASE_HOST"],
        port=os.environ["DATABASE_PORT"],
        database=os.environ["DATABASE_NAME"],
        user=os.environ["DATABASE_USER"],
        password=os.environ["DATABASE_PASS"],
    )
