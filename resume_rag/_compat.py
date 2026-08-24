"""Compatibility shim for RAGAS 0.4.x importing a removed langchain-community path.

RAGAS does `from langchain_community.chat_models.vertexai import ChatVertexAI`
at import time, but that module was removed from current langchain-community.
We register a stub module so the import resolves. ChatVertexAI is never used
(we run RAGAS with our own Claude LLM), so a placeholder is sufficient.

Import this BEFORE importing ragas anywhere.
"""

from __future__ import annotations

import sys
import types


def install_vertexai_shim() -> None:
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return  # real module exists; no shim needed
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(mod_name)

    class ChatVertexAI:  # placeholder — never instantiated in our pipeline
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ChatVertexAI shim — Resume-RAG does not use Vertex AI."
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = shim