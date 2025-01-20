"""Main entry point of the RAG application backend.

It exposes all of the REST api that must be made available to an application
developer willing to conduct semantic search & indexation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import onnxruntime_genai as og
import pymupdf
import pymupdf4llm
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .database import RelevantDocument, find_similar_to, is_already_indexed, save_chunks
from .preprocessing import preprocess

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

app = FastAPI()

###### LANGUAGE MODELS ########################################################
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED: SentenceTransformer = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=DEVICE)

def ort_llm_model() -> og.Model:
    """Return the appropriate ORT-genai model."""
    path: str = "onnx/llama-3.2-1b-instruct__int4"
    if torch.cuda.is_available():
        path  = "onnx/llama-3.2-1b-instruct__int4"
    return og.Model(path)

def ort_llm_respond(model: og.Model, user_prompt: str, new_tokens: int) -> Iterator[str]:
    """Return the text response from the llm using a quantized llm."""
    tokenizer: og.Tokenizer = og.Tokenizer(model)
    streamer: og.TokenizerStream = tokenizer.create_stream()
    prompt: str = f"<|user|>\n{user_prompt} <|end|>\n<|assistant|>## Answer\n"
    tokens: list[int] = tokenizer.encode(prompt)
    # generation config
    params: og.GeneratorParams = og.GeneratorParams(model)
    params.set_search_options(
        max_length = len(tokens) + new_tokens,
        temperature = 0.05,
        top_k = 5,
        top_p = 0.95,
        repetition_penalty = 1.1,
    )
    params.input_ids = tokens
    generator: og.Generator = og.Generator(model, params)
    # generation loop
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        yield streamer.decode(generator.get_next_tokens()[0])

def ort_pipeline() -> Callable[[str], Iterator[str]]:
    """Pipeline to generate text response to a prompt."""
    model = ort_llm_model()
    return lambda text, new_tokens: ort_llm_respond(model, text, new_tokens)

def hf_llm_model() -> AutoModelForCausalLM:
    """Return the huggingface (pytorch) model for the generation model."""
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def hf_llm_respond(model: AutoModelForCausalLM, user_prompt: str, new_tokens: int) -> Iterator[str]:
    """Return the text response from the llm."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    streamer  = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt = f"<|user|>\n{user_prompt} <|end|>\n<|assistant|>## Answer\n"
    tokens = tokenizer(prompt, return_tensors='pt').to(model.device)
    _outputs = model.generate(
        **tokens,
        streamer=streamer,
        max_new_tokens=new_tokens,
        do_sample=True,
        temperature=0.05,
        top_k=5,
        top_p=0.95,
    )
    yield from streamer

def hf_pipeline() -> Callable[[str], Iterator[str]]:
    """Pipeline to generate text response to a prompt."""
    model = hf_llm_model()
    return lambda text, new_tokens: hf_llm_respond(model, text, new_tokens)

GEN = hf_pipeline()

###### UTILITY CLASSES ########################################################
class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray

class SearchRequest(BaseModel):
    """An user request expressed in natural language to retrieve user documents."""

    question: str
    search_results: int = 5

class GenerationRequest(BaseModel):
    """A request to generate a plaintext response for a given prompt."""

    prompt: str
    generation_tokens: int = 256

class RagRequest(BaseModel):
    """An user request asked to the system using natural language."""

    question: str
    search_results: int = 5
    generation_tokens: int = 256
    append_sources: bool = True


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
async def retrieve(request: SearchRequest) -> list[RelevantDocument]:
    """Retrieve the `nb_results` most relevant documents based on a user request."""
    prompt: str = f"""Given a user request, attempt to find the most relevant
    piece of information in order to satisfy that user's request.

    # User Request:
    {request.question}
    """
    # just drop the leading whitespaces in the prompt.
    prompt: str = "\n".join(line.strip() for line in prompt.splitlines())
    return find_similar_to(EMBED.encode(prompt), request.search_results)

@app.post("/rag")
async def rag(req: RagRequest) -> StreamingResponse:
    """Perform a complete rag pipeline: it starts from a user requests and generates an answer."""
    documents: list[RelevantDocument] = await retrieve(SearchRequest(
        question=req.question,
        search_results=req.search_results,
    ))
    sources: str = "\n".join(
        f"Ref: [{i}]\nDocument: **{d.path_to_doc}**\n```\n{d.text}```" for i,d in enumerate(documents)
    )
    prompt: str = f"""
    Use information from the given documents to answer the question below.
    Be succinct in your response.
    When unsure about the answer, express your doubts clearly.

    ## Question
    {req.question}

    ## Archives
    {sources}
    """
    prompt: str = "\n".join(line.strip() for line in prompt.splitlines())

    def _respond() -> Iterator[str]:
        """Generate response chunk by chunk."""
        yield from GEN(prompt, req.generation_tokens)
        if req.append_sources:
            yield "\n\n\n\n"
            yield "## Sources\n"
            yield sources

    return StreamingResponse(_respond(), media_type="text/plain")

@app.post("/generation/")
async def generate(request: GenerationRequest) -> StreamingResponse:
    """Generate a text response to the given prompt."""
    return StreamingResponse(GEN(request.prompt, request.generation_tokens), media_type="text/plain")#GEN([{"role": "user", "content": prompt}], max_new_tokens=512)[0]["generated_text"][-1]["content"]


async def get_md_text(file: UploadFile) -> str:
    """Extract text from the given file in markdown."""
    return pymupdf4llm.to_markdown(pymupdf.open(stream=await file.read(), filename=file.filename), show_progress=False)

async def get_text(file: UploadFile) -> str:
    """Extract text from the given file without using ocr."""
    return "\n".join([page.get_text() for page in pymupdf.open(stream=await file.read(), filename=file.filename)])


def get_ocr_text(file: UploadFile) -> str:
    """Extract text from the given file and use ocr as needed."""
    return "\n".join([page.get_text(textpage=page.get_textpage_ocr()) for page in pymupdf.open(file.file)])
