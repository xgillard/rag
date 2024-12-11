# This image contains all the required dependencies to create a hardware
# accelerated RAG backend server. It comes with the following packages
# preinstalled:
# - NVIDIA CUDA drivers 12.4
# - Python3
# - Pytorch
# - Tesseract-OCR
# - Transformers, Accelerate (Huggingface)
# - FastAPI
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Step 1: install tesseract-ocr along with the support for various 
# languages so as to make it available to pymupdf. This way, tesseract
# can be used to acquire text data from documents comprising scans of
# text.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-nld \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-spa

# Step 2: install the necessary python packages. Basically:
# - torch, torchvision, torchdata
# - transformers accelerate
# - pymupdf
# - pyscopg 
# - fastapi
RUN python3 -m venv .venv
RUN pip3 install --upgrade pip
RUN pip3 install psycopg2-binary
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip3 install --requirement /tmp/requirements.txt

# Step 3: Export the ad-hoc environment variable to make sure tesseract is
# made available to pymupdf.
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# In order to run this container interactively with hardware acceleration
# enabled, just run the following command:
# ```
# docker run -it --gpus=all xgillard/rag-base
# ```
