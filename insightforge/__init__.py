"""InsightForge - an AI-powered Business Intelligence Assistant.

Capstone project for the *Advanced Generative AI* programme.

The package is organised along the seven steps of the problem statement::

    1. Data preparation .............. data_loader.py
    2. Knowledge base creation ....... knowledge_base.py
    3. LLM application development ... analysis.py + retriever.py + llm.py
    4. Chain prompts ................. prompts.py + rag_chain.py
    5. RAG system setup .............. rag_chain.py
    6. Memory integration ............ rag_chain.py (ConversationBufferMemory)
    7. LLMOps ........................ evaluation.py, monitoring.py,
                                       visualization.py, ../app.py
"""

from insightforge.config import settings

__all__ = ["settings", "__version__"]
__version__ = "1.0.0"
