# Backend

The backend of the RAG process is where most of the magic happens.
This container is in charge of exposing the core functionalities of the 
appliaction such as: document indexation, retrieval and answer generation.

**Remark**
If you are a maintainer of the frontend and you dont know what API are 
offered by this container nor how to use them; please note that a complete
__OpenAPI 3.0__ documentation of the services is made available to you
at `http://wherever_this_container_runs/docs`.

## Technicalities

All REST API are written in python with the `fastapi` library. By default
the container exposes these API on port `tcp:80`. The text extraction 
from documents is carried out using `pymupdf`. 

**At this point, it was decided not to use the OCR capabilities of that lib to extract document data**

### Embedding Computation

Because it does not suffice to just use the last hidden state from any 
pretrained model to compute text embeddings (one should actually use
pretrained sentence transformer models to compute the embedding of a 
full text), I have chosen to pick: 
[`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) 
as an embedding model. 

This choice was simply motivated by the fact that this is currently one
of the best available models according to the 
[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

This 'one of the best' arguments is however subject to the following
criteria: 
1. The model must be small (less than 1B params) because we do not have
   a lot of compute power that could be devoted to this embedding task.
2. The model must be open source (`jina-embeddings-v3` is licensed under 
   the terms of the "CC BY-NC 4.0" license).
3. The model has to be multilingual so as to be able to cope with documents
   redacted in different languages and answer to requests that are written
   in many different languages.

#### Embedding Post processing

All embeddings are pooled using *average pooling* (but I implemented
*max pooling* and *cls-token pooling* strategies as well). They are 
further normalized to a magnitude of one with L2 normalisation 
(all embeddings are divided by their L2-norm aka size in a generalized
euclidian space).