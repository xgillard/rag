"""Commonalities for working with embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypeVar

import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .preprocessing import preprocess

if TYPE_CHECKING:
    import numpy as np


class EmbeddedChunk(NamedTuple):
    """A document chunk with its embedding."""

    text: str
    embedding: np.ndarray


class EmbeddingPipeline:
    """An utility class to compute the embeddings for queries and documents.

    This pipeline is meant to compute the queries and documents embeddings
    in order to retrieve and or rerank chunks of documents from the corpus
    that is stored in the vector database.

    # Note:
    One might ask: "what is the added value of using this class rather than
    huggingface's provided 'feature-extraction' pipeline ?". This would be
    a good question. And the answer is twofolds:
    1. it computes embeddings differently for the query than the document
       (althought both use average pooling, the query produces one single
       embedding whereas the document version computes one embedding per
       chunk).
    2. it chunks the text and does not truncate the input even when that input
       is very long.
    """

    model: AutoModel
    tokenizer: AutoTokenizer
    stride: int = 20
    max_length: int = 512

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: torch.device | str = "cpu",
        stride: int = 20,
        max_length: int = 512,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the pipeline.

        The model and tokenizer are automatically fetched from huggingface hub
        based on the checkpoint name. If any additional kwargs (typically the
        `torch_dtype` and/or `trust_remote_code` flag) need to be passed to
        AutoModel while creating the model, then these params can be passed
        on via the `kwargs`.

        ## Parameters
        - pretrained_model_name_or_path: the name (str) of the model checkpoint
                    on huggingface hub. For instance: 'jinaai/jina-embeddings-v3'.
        - device: (optional torch.device or str) the device where the model must
                  be located for before executing. Typically, if you have one
                  nvidia GPU available and your torch installation is CUDA enabled,
                  then the value of that parameter will be "cuda".
        - stride: (int) the 'overlap' between any two windows of tokenized text in
                  case a text sample (document or query) is too long to be processed
                  in one go by the underlying model.
        - max_length: (int) the number of tokens that should be kept in one chunk
                  of text when the text of a document is so long that it must be
                  split in several chunks.
        - kwargs: any additional params you might want to pass on to the model for
                  its instantiation. For example: `trust_remote_code=True` if the
                   model you intend to use requires additional unvetted code for
                   its execution or;  `torch_dtype=torch.bfloat16` if you want to
                   use brainfloat (half precision).
        """
        self.device = device
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.stride = stride
        self.max_length = max_length

    def encode_document(self, text: str) -> list[EmbeddedChunk]:
        """Compute the embeddings for the various chunks of a document.

        The document is 1st chunked by the tokenizer and then document
        embedding is computed for each chunk.
        """
        text: str = self.preprocess(text)
        encoded: dict[str, Any] = self.tokenize(text)
        embed: torch.tensor = self.forward(encoded)  # shape [b, s, h]
        embed: torch.tensor = EmbeddingPipeline.avg_pool_multi(embed, encoded["attention_mask"])  # shape: [b, h]
        embed: torch.tensor = EmbeddingPipeline.normalize(embed)  # shape: [b, h]
        embed: np.ndarray = embed.to(torch.float32).cpu().numpy()
        return [
            EmbeddedChunk(EmbeddingPipeline.original_text(text, encoded, i), embed[i, :]) for i in range(len(embed))
        ]

    def encode_query(self, text: str) -> np.ndarray:
        """Compute one single embedding meant to represent the whole user query.

        In the event where the user query is very long, the tokenizer will split
        it in several contexts and the model will produce one embedding for each
        chunk. These many embeddings, however, are not really useful when it comes
        to retrieving relevant documents to answer the query. This is why, in the
        event of a very long query, the chunks embeddings are pooled together so
        as to produce one single vector to encode the user request.
        """
        text: str = self.preprocess(text)
        encoded: dict[str, Any] = self.tokenize(text)
        embed: torch.tensor = self.forward(encoded)  # shape [b, s, h]
        embed: torch.tensor = EmbeddingPipeline.avg_pool_single(embed, encoded["attention_mask"])  # shape: [h, ]
        embed: torch.tensor = EmbeddingPipeline.normalize(embed)  # shape: [h, ]
        return embed.to(torch.float32).cpu().numpy()

    def preprocess(self, text: str) -> str:
        """Apply some preprocessing to the text.

        This step is meant to facilitate reconciliation between queries and documents.
        For instance, this step replaces all esoteric characters by some ASCII standard
        characters so as to have a 'normalized', 'clear' representation of the given
        text.

        Unless this method is overridden, the preprocessing happens according to the
        rules defined by the 'preprocessing.preprocessing' function.
        """
        return preprocess(text)

    def tokenize(self, text: str) -> dict[str, Any]:
        """Chunk the text if need be and return an a BatchEncoding.

        That is, it returns a dict-like object comprising the elements that must be
        passed to the forward method in order to perform the actual feature extraction.

        ## In most cases
        (understand, this should always be the case, unless you have overridden both
        methods `tokenize` and `forward`) This method must return a dict like
        object comprising a least the following three keys:

        - input_ids: (torch.tensor of shape [b, s]) a batch of tokenized sequences
              that must go through the transformer in order to perform the feature
              extraction. THIS TENSOR MUST BE PLACED ON THE SAME DEVICE AS `self.model`
              THAT IS, IT MUST BE PLACED ON `self.device`.
        - attention_mask: (torch.tensor of shape [b, s]) the attention mask corresponding
              to each entry in the given batch of sequence. This mask is used both as an
              input parameter to the model and to compute an appropriately weighed
              final embedding. THIS TENSOR MUST BE PLACED ON THE SAME DEVICE AS `self.model`
              THAT IS, IT MUST BE PLACED ON `self.device`.
        - offset_mapping: (torch.tensor of shape [b, s, 2]) a tensor comprising the start
              and stop position corresponding to each token in the input text. This piece
              of information is used to retrieve the original text corresponding to each
              chunk of encoded text.
        """
        encoded: BatchEncoding = self.tokenizer(
            text,
            stride=self.stride,
            return_overflowing_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "offset_mapping": encoded["offset_mapping"],
        }

    @torch.no_grad()
    def forward(self, encoded: dict[str, Any]) -> torch.tensor:
        """Compute a model representation for each sample in the batch.

        The shape of the return tensor is [b, s, h] which means, it returns one
        hidden representation per sample in the batch of inputs. Typically, this
        corresponds to the last hidden state of the model (but you may choose to
        do fancy stuffs if you so like.)

        # Parameters:
        - task: (str) a string param that can be used to tune the model behavior
            so as to customize the coloration of the embeddings that are produced.
        - encoded: (dict) this dictionary contains serveral tensors that can be
            passed on to the model in order to produce the hidden representation.
            Typically, this dictionnary should comprise tensor values for (at least)
            the following piece of information: 'input_ids', 'attention_mask',
            'offset_mapping'. See `self.tokenize` for further information about that
            dict.
        """
        return self.model(encoded["input_ids"], encoded["attention_mask"]).last_hidden_state.detach()

    @staticmethod
    def avg_pool_multi(last_hidden_state: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """Perform a pass of average pooling on the given input tensor to produce one embedding per sequence."""
        # unsqueeze to get a tensor of shape [b, s, 1]
        # expand bcast to  a tensor of shape [b, s, h].
        am: torch.tensor = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        embed = (last_hidden_state * am).sum(1)
        mask = am.sum(1)
        return (embed / mask).detach()

    @staticmethod
    def avg_pool_single(last_hidden_state: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """Perform a pass of average pooling on the given input tensor to produce one single embedding."""
        # unsqueeze to get a tensor of shape [b, s, 1]
        # expand bcast to  a tensor of shape [b, s, h].
        b, s, h = last_hidden_state.shape
        am: torch.tensor = attention_mask.unsqueeze(-1).expand((b, s, h))
        embed = (last_hidden_state * am).view((-1, h)).sum(0)
        mask = am.view((-1, h)).sum(0)
        return (embed / mask).squeeze()  # shape [h,]

    @staticmethod
    def normalize(embedding: torch.tensor) -> torch.tensor:
        """Normalize the magnitude of the embedding tensor (L2 normalization)."""
        return embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)

    @staticmethod
    def original_text(text: str, encoded: BatchEncoding, i: int) -> str:
        """Return the original text for the ith sample in the batch encoding."""
        offsets: torch.tensor = encoded["offset_mapping"][i]
        start: int = offsets[1, 0]  # (1,0) because (0,0) is the special [CLS] token
        stop: int = offsets.max()
        return text[start : stop + 1]


#############################################################################################
# BEYOND THIS LINE ARE SOME CUSTOM PIPELINES THAT CUSTOMIZE THE GENERIC PIPELINE WITH THEIR #
# OWN SPECIFIC REQUIREMENTS. NAMELY, THE FOLLOWING LINES PROVIDE PIPELINES IMPLEMENTATIONS  #
# FOR THE FOLLOWING MODELS:                                                                 #
# - jinaai/jina-embeddings-v3                                                               #
# - intfloat/multilingual-e5-large-instruct                                                 #
#############################################################################################

T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """Implement a singleton pattern for the decorated class."""
    __instances = {}

    def getinstance(*args: ..., **kwargs: dict[str, Any]) -> T:
        """Create an instance with the given parameters."""
        if cls not in __instances:
            __instances[cls] = cls(*args, **kwargs)
        return __instances[cls]

    return getinstance


@singleton
class MultilingualE5InstructPipeline(EmbeddingPipeline):
    """Customize the EmbeddingPipeline to use intfloat/multilingual-e5-large-instruct.

    # Note:
    This class is meant to be a singleton object in order to not overload the available GPU
    with too many instances of the same models.
    """

    CHECKPT: str = "intfloat/multilingual-e5-large-instruct"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.float16
    # intfloat/multilingual-e5-large-instruct is an instruct model.
    # For asymmetric tasks such as retrieval, it is necessary to prepend a prompt to the
    # query so that the model can adapt its weight 'coloration' to the task at hannd.
    PROMPT: str = """Given a user request, attempt to find the most relevant
    piece of information in order to satisfy that user's request.

    # User Request:
    """

    def __init__(self) -> None:
        """Initialize the singleton instance."""
        super().__init__(self.CHECKPT, self.DEVICE, torch_dtype=self.DTYPE)

    # @override
    def encode_query(self, text: str) -> np.ndarray:
        """Override EmbeddingPipeline.encode_query."""
        text: str = self.PROMPT + text
        return super().encode_query(text)


@singleton
class JinaaiPipeline(EmbeddingPipeline):
    """Customize the EmbeddingPipeline to work with 'jinaai/jina-embeddings-v3'."""

    CHECKPT: str = "jinaai/jina-embeddings-v3"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    def __init__(self) -> None:
        """Initialize the singleton instance."""
        super().__init__(self.CHECKPT, self.DEVICE, torch_dtype=self.DTYPE, trust_remote_code=True)

    # @override
    def encode_document(self, text: str) -> list[EmbeddedChunk]:
        """Override EmbeddingPipeline.encode_document."""
        text: str = self.preprocess(text)
        encoded: dict[str, Any] = self.tokenize(text)
        embed: torch.tensor = self.forward("retrieval.passage", encoded)  # shape [b, s, h]
        embed: torch.tensor = EmbeddingPipeline.avg_pool_multi(embed, encoded["attention_mask"])  # shape: [b, h]
        embed: torch.tensor = EmbeddingPipeline.normalize(embed)  # shape: [b, h]
        embed: np.ndarray = embed.to(torch.float32).cpu().numpy()
        return [
            EmbeddedChunk(EmbeddingPipeline.original_text(text, encoded, i), embed[i, :]) for i in range(len(embed))
        ]

    # @override
    def encode_query(self, text: str) -> np.ndarray:
        """Override EmbeddingPipeline.encode_query."""
        text: str = self.preprocess(text)
        encoded: dict[str, Any] = self.tokenize(text)
        embed: torch.tensor = self.forward("retrieval.query", encoded)  # shape [b, s, h]
        embed: torch.tensor = EmbeddingPipeline.avg_pool_single(embed, encoded["attention_mask"])  # shape: [h, ]
        embed: torch.tensor = EmbeddingPipeline.normalize(embed)  # shape: [h, ]
        return embed.to(torch.float32).cpu().numpy()

    # @override
    @torch.no_grad()
    def forward(self, task: str, encoded: dict[str, Any]) -> torch.tensor:
        """Override EmbeddingPipeline.forward."""
        input_ids: torch.tensor = encoded["input_ids"]
        attn_mask: torch.tensor = encoded["attention_mask"]
        task_id: int = self.model._adaptation_map[str(task)]  # noqa: SLF001
        adapter_mask: torch.tensor = torch.full((input_ids.shape[0],), task_id, dtype=torch.int32)
        return self.model(input_ids, attn_mask, adapter_mask=adapter_mask).last_hidden_state.detach()
