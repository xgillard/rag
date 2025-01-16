"""Main entry point of the RAG application backend.

It exposes all of the REST api that must be made available to an application
developer willing to conduct semantic search & indexation.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, NamedTuple

import pymupdf
import pymupdf4llm
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .database import RelevantDocument, find_similar_to, is_already_indexed, save_chunks
from .preprocessing import preprocess

if TYPE_CHECKING:
    import numpy as np

app = FastAPI()

###### LANGUAGE MODELS ########################################################
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED: SentenceTransformer = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=DEVICE)
GEN  = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map="auto")

###### UTILITY CLASSES ########################################################
class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray


###### SERVICE ENDPOINTS ######################################################
@app.post("/chunks/")
async def get_chunks(file: UploadFile) -> list[str] | None:
    """Return the list of paragraphs in the document."""
    try:
        text: str = await get_md_text(file)
        result: list[str] = []

        acc: str = ""
        for p in text.split("\n\n"):
            stripped: str = p.strip()
            if len(acc) > 600 and len(stripped) > 100:  # noqa: PLR2004 # environ 10 lignes de texte
                result.append(preprocess(acc))
                acc = stripped
            else:
                acc += "\n\n" + stripped
        if acc:
            result.append(preprocess(acc))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"FAIL : chunking -- {e}")  # noqa: B904
    else:
        return result


@app.post("/indexation/")
async def index_file(path_to_doc: str, file: UploadFile) -> str | None:
    """Perform the indexation of one single file sent over http."""
    try:
        if cnt := is_already_indexed(path_to_doc):
            return f"SKIP {path_to_doc} -- {cnt}"

        paragraphs: list[str] = await get_chunks(file)
        embeddings: np.ndarray = EMBED.encode(paragraphs)

        chunks: list[EmbeddedChunk] = [EmbeddedChunk(text=p, embedding=embeddings[i]) for i,p in enumerate(paragraphs)]
        save_chunks(path_to_doc, chunks)
        return f"DONE : {path_to_doc}"  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"FAIL : {path_to_doc} -- {e}")  # noqa: B904
    return None


@app.post("/retrieval/")
async def retrieve(user_request: str, nb_results: int = 5) -> list[RelevantDocument]:
    """Retrieve the `nb_results` most relevant documents based on a user request."""
    prompt: str = f"""Given a user request, attempt to find the most relevant
    piece of information in order to satisfy that user's request.

    # User Request:
    {user_request}
    """
    # just drop the leading whitespaces in the prompt.
    prompt: str = "\n".join(line.strip() for line in prompt.splitlines())
    return find_similar_to(EMBED.encode(prompt), nb_results)

@app.post("/rag")
async def rag(question: str) -> str:
    """Perform a complete rag pipeline: it starts from a user requests and generates an answer."""
    documents: list[RelevantDocument] = await retrieve(question, nb_results=5)
    sources: str = "\n".join(
        f"{'#'*40}\n{d.path_to_doc}\n{'-'*40}\n{d.text}" for d in documents
    )
    prompt: str = f"""
    You are an assistant to the university archivists. Reply to the archivist's
    question as good as possible using only the (possibly relevant) pieces of archive
    below. When an answer cannot be found in the given archives,
    just say "SORRY, I COULD NOT FIND THE ANSWER IN THE PROVIDED ACHIVES".
    Do not forget to provide links to the sources you've used at the end of your answer.

    ## Archives
    {sources}

    ## Archivist's Question
    {question}
    """
    prompt: str = "\n".join(line.strip() for line in prompt.splitlines())
    return await generate(prompt)

@app.post("/generation/")
async def generate(prompt: str) -> str:
    """Generate a text response to the given prompt."""
    return GEN([{"role": "user", "content": prompt}], max_new_tokens=512)[0]["generated_text"][-1]["content"]


async def get_md_text(file: UploadFile) -> str:
    """Extract text from the given file in markdown."""
    return pymupdf4llm.to_markdown(pymupdf.open(stream=await file.read(), filename=file.filename), show_progress=False)

async def get_text(file: UploadFile) -> str:
    """Extract text from the given file without using ocr."""
    return "\n".join([page.get_text() for page in pymupdf.open(stream=await file.read(), filename=file.filename)])


def get_ocr_text(file: UploadFile) -> str:
    """Extract text from the given file and use ocr as needed."""
    return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(file.file)])
