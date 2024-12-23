"""Main entry point of the RAG application backend.

It exposes all of the REST api that must be made available to an application
developer willing to conduct semantic search & indexation.
"""

from __future__ import annotations

import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile

from .database import RelevantDocument, find_similar_to, is_already_indexed, save_chunks
from .embedding import EmbeddedChunk, MultilingualE5Pipeline
from .generation import GenerationPipeline

app = FastAPI()

PIPE = MultilingualE5Pipeline()
GEN  = GenerationPipeline()

@app.post("/indexation/")
async def index_file(path_to_doc: str, file: UploadFile) -> str | None:
    """Perform the indexation of one single file sent over http."""
    try:
        if cnt := is_already_indexed(path_to_doc):
            return f"SKIP {path_to_doc} -- {cnt}"

        text: str = await get_text(file)
        chunks: list[EmbeddedChunk] = PIPE.encode_document(text)
        save_chunks(path_to_doc, chunks)
        return f"DONE : {path_to_doc}"  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"FAIL : {path_to_doc} -- {e}")  # noqa: B904
    return None


@app.post("/retrieval/")
async def retrieve(user_request: str, nb_results: int = 5) -> list[RelevantDocument]:
    """Retrieve the `nb_results` most relevant documents based on a user request."""
    return find_similar_to(PIPE.encode_query(user_request), nb_results)


@app.post("/generation/")
async def generate(prompt: str) -> str:
    """Generate a text response to the given prompt."""
    return GEN([{"role": "user", "content": prompt}])

async def get_text(file: UploadFile) -> str:
    """Extract text from the given file without using ocr."""
    return "\n".join([page.get_text() for page in pymupdf.open(stream=await file.read(), filename=file.filename)])


def get_ocr_text(file: UploadFile) -> str:
    """Extract text from the given file and use ocr as needed."""
    return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(file.file)])
