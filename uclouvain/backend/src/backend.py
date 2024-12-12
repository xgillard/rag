"""Main entry point of the RAG application backend.

It exposes all of the REST api that must be made available to an application
developer willing to conduct semantic search & indexation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psycopg2
import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .embedding import EmbeddedChunk, encode_document, encode_query

if TYPE_CHECKING:
    import numpy as np

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
            text: str = await get_text(file)
            chunks: list[EmbeddedChunk] = encode_document(text)

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


class RelevantDocument(BaseModel):
    """One model that is deemed relevant for the user request."""

    path_to_doc: str
    text: str


@app.post("/retrieval/")
async def retrieve(user_request: str, nb_results: int = 5) -> list[RelevantDocument]:
    """Retrieve the `nb_results` most relevant documents based on a user request."""
    embed: np.ndarray = encode_query(user_request)

    with get_database().cursor() as cur:
        param: str = ", ".join(str(x) for x in embed.tolist())
        param: str = f"[{param}]"
        cur.execute(
            """
            SELECT path_to_doc, text
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (param, nb_results),
        )
        return [RelevantDocument(path_to_doc=r[0], text=r[1]) for r in cur.fetchall()]


async def get_text(file: UploadFile) -> str:
    """Extract text from the given file without using ocr."""
    return "\n".join([page.get_text() for page in pymupdf.open(stream=await file.read(), filename=file.filename)])


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
