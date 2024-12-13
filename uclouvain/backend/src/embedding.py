"""Commonalities for working with embeddings."""

from typing import NamedTuple

import numpy as np


class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray
