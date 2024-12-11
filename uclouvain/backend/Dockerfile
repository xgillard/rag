# This is the docker container image for the semantic search (RAG) backend.
# 
# The definition of this contained is based on the `rag-base` image 
# (see ../rag-base) to keep this definition streamlined and cut build times.
#
# Essentially, this container provides the REST API written in python 
# (fastapi) that provide the core functionality of the semantic search engine:
# indexation, retrieval and and answer generation.
FROM xaviergillard/rag-base

RUN --mount=type=bind,source=prefetch.py,target=/tmp/prefetch.py \
    python /tmp/prefetch.py

# Install the application files in the container
COPY src/* /workspace/

# And serve them with fastapi for production setup. 
# Please note that this container exposes it services on port tcp:80, so
# make sure to remap that port to whatever you would like to expose. 
CMD [ "fastapi", "run", "backend.py", "--port", "80"]