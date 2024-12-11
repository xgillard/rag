# RAG-BASE

This image does not per-se belong to the semantic search project I am pursuing.
It is however used as the base image for the backend server. Indeed, this 
image is meant to be a reusable (without rebuilding) accross several RAG 
projects.

This image is derived from `pytorch/pytorch` and provides all the necessary
dependencies to prop up a hardware accelerated platform suitable to build
a RAG backend (indexation and so on...).

Essentially, it provides an image with:
* CUDA drivers
* Pytorch 2.5.1
* Python3
* Tesseract-OCR
* Transformers, Accelerate (Huggingface)
* PymuPDF (with a preconfigured tesseract)
* FastAPI

## Building the image

Nothing fancy is required to build this image: just

```
docker build -t xaviergillard/rag-base .
```

In order to run this image with hardware acceleration enabled, just launch
it as follows:

```
docker run -it --gpus=all xaviergillard/rag-base
```

It may however be even better to include a contained based of this image in
a docker-compose, and configure it as follows:

```yaml
  backend:
    image: xaviergillard/rag-base
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
```