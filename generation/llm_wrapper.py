"""
generation/llm_wrapper.py
────────────────────────────────────────────────────────────────────────────────
Clean abstraction for LLM backends (Mock, Groq, OpenAI, HuggingFace).

Classes:
    BaseLLM     — Abstract interface
    MockLLM     — Simulator for rapid testing (free, no GPU)
    GroqLLM     — Wrapper for Groq API (FREE, extremely fast LPU inference)
    OpenAILLM   — Wrapper for OpenAI API
    HFLocalLLM  — Wrapper for local HuggingFace models (transformers)

Recommended default for journal work: GroqLLM (Llama 3.3 70B, free tier)
    → Set GROQ_API_KEY env variable or add to .env file
    → Get free key at: https://console.groq.com

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import os
import time
import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from utils.logger import logger


# ════════════════════════════════════════════════════════════════════════════
class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass


# ════════════════════════════════════════════════════════════════════════════
class MockLLM(BaseLLM):
    """
    Simulator for testing the RAG pipeline without real LLM costs or hardware.
    Mocks structured output (Confidence, Explanation, Answer).
    """

    def generate(self, prompt: str, **kwargs) -> str:
        # Simulate local latency
        time.sleep(random.uniform(0.1, 0.3))

        # Check if context is "empty"
        if "I don't know" in prompt.lower() and "Context:" in prompt and len(prompt.split("Context:")[1].strip()) < 10:
             return "Confidence: 0.1\nExplanation: No relevant context found.\nAnswer: I don't know."

        # Randomly decide if confident
        conf     = round(random.uniform(0.3, 0.95), 2)
        ans_mock = "Based on the provided documents, the answer is related to the query but synthesized by this mock model."

        return (
            f"Confidence: {conf}\n"
            f"Explanation: The retrieved documents explicitly mention the key entities.\n"
            f"Answer: {ans_mock}"
        )


# ════════════════════════════════════════════════════════════════════════════
class GroqLLM(BaseLLM):
    """
    Wrapper for Groq Cloud API — FREE tier, extremely fast LPU inference.
    Default model: llama-3.3-70b-versatile (best free option for RAG).

    Setup:
        pip install groq
        Set GROQ_API_KEY in environment or .env file.
        Get a free API key at: https://console.groq.com
    """

    def __init__(self, model_id: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")

        self.model_id = model_id
        self.api_key  = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get your free key at https://console.groq.com\n"
                "Then run: set GROQ_API_KEY=your_key_here (Windows)"
            )
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Groq client initialised  |  model={self.model_id}")

    def generate(self, prompt: str, **kwargs) -> str:
        temp       = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_new_tokens", 512)

        response = self.client.chat.completions.create(
            model    = self.model_id,
            messages = [{"role": "user", "content": prompt}],
            temperature = temp,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content


# ════════════════════════════════════════════════════════════════════════════
class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI Chat Completion API."""

    def __init__(self, model_id: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        self.model_id = model_id
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        temp = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_new_tokens", 512)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ════════════════════════════════════════════════════════════════════════════
class HFLocalLLM(BaseLLM):
    """Wrapper for local models via HuggingFace transformers."""

    def __init__(self, model_id: str, device: str = "cpu"):
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        logger.info(f"Loading local HF model: {model_id} on {device} …")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        )

    def generate(self, prompt: str, **kwargs) -> str:
        temp = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_new_tokens", 512)

        out = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True if temp > 0 else False,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        # Extract only the generated part
        full_text = out[0]["generated_text"]
        return full_text[len(prompt):].strip()


# ════════════════════════════════════════════════════════════════════════════
def get_llm_backend(cfg: dict) -> BaseLLM:
    """Factory to get the correct LLM backend based on config."""
    g_cfg = cfg.get("generation", {})
    type_ = g_cfg.get("model", "mock").lower()
    m_id  = g_cfg.get("model_id", "gpt-3.5-turbo")

    if type_ == "mock":
        logger.info("Using Mock LLM backend (Simulator).")
        return MockLLM()
    elif type_ == "groq":
        logger.info(f"Using Groq backend: {m_id}")
        return GroqLLM(model_id=m_id)
    elif type_ == "openai":
        logger.info(f"Using OpenAI backend: {m_id}")
        return OpenAILLM(model_id=m_id)
    elif type_ == "huggingface":
        logger.info(f"Using Local HF backend: {m_id}")
        return HFLocalLLM(model_id=m_id, device="cpu")
    else:
        raise ValueError(f"Unknown LLM backend: '{type_}'. Choose from: mock, groq, openai, huggingface")
