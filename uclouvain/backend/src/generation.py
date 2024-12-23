"""Utilities to generate text responses to user prompt."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerationPipeline:
    """Generation pipeline."""

    CHECKPOINT = "meta-llama/Llama-3.2-1B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    @torch.no_grad()
    def __init__(self) -> None:
        """Initialize the generation pipeline."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.CHECKPOINT,
            torch_dtype=self.DTYPE,
            ).to(self.DEVICE).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.CHECKPOINT)

    def __call__(self, messages: list[dict[str, str]]) -> str:
        """Generate text in response to a user prompt."""
        template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            ).to(self.DEVICE)
        response = self.model.generate(template, max_new_tokens=2048)
        response = self.tokenizer.decode(response[0])
        return response  # noqa: RET504
