"""Commonalities for working with the database."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psycopg2
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np

    from .embedding import EmbeddedChunk


class RelevantDocument(BaseModel):
    """One model that is deemed relevant for the user request."""

    path_to_doc: str
    text: str


def get_database() -> psycopg2.connection:
    """Open a connection to the database."""
    return psycopg2.connect(
        host=os.environ["DATABASE_HOST"],
        port=os.environ["DATABASE_PORT"],
        database=os.environ["DATABASE_NAME"],
        user=os.environ["DATABASE_USER"],
        password=os.environ["DATABASE_PASS"],
    )


def is_already_indexed(uri: str) -> bool:
    """Return True iff the file is already known in the database."""
    conn = get_database()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(text) FROM documents WHERE path_to_doc = %s;", (f"{uri}",))
        cnt: int = cur.fetchone()[0]
        return cnt > 0


def save_chunks(path_to_doc: str, chunks: list[EmbeddedChunk]) -> None:
    """Save the chunks in the database."""
    conn = get_database()
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "INSERT INTO documents(path_to_doc, text, embedding) VALUES(%s, %s, %s);",
                (f"{path_to_doc}", chunk.text, chunk.embedding.tolist()),
            )
        conn.commit()


def find_similar_to(embedding: np.ndarray, limit: int = 100) -> list[RelevantDocument]:
    """Find relevant documents similar to the given embedding."""
    param: str = ", ".join(str(x) for x in embedding.tolist())
    param: str = f"[{param}]"
    with get_database().cursor() as cur:
        cur.execute(
            """
            SELECT path_to_doc, text
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (param, limit),
        )
        return [RelevantDocument(path_to_doc=r[0], text=r[1]) for r in cur.fetchall()]
