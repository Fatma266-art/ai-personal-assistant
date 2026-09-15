"""
LLM Reasoning Module
--------------------
This module takes a user query, retrieves relevant context from the vector store,
and uses an LLM to understand the request and generate a reasoned response.
"""

from pathlib import Path
import logging
import os
import sys
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path so we can import from retrieval
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retrieve import load_retriever, retrieve

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Settings
LLM_MODEL = os.getenv("LLM_MODEL", "phi3")   
TOP_K = int(os.getenv("TOP_K", "3"))


def get_openai_client() -> OpenAI:
    """Initialize client that talks to Groq's hosted API."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Get one free from console.groq.com"
        )

    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )


def build_context(results: List[Dict]) -> str:
    """Turn retrieved chunks into a clean context string for the LLM."""
    if not results:
        return "No relevant context found."

    context_parts = []
    for r in results:
        source = r.get("source", "unknown")
        text = r.get("text", "")
        context_parts.append(f"[Source: {source}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the LLM.
    The system prompt teaches the model to reason about the user request.
    """
    system_prompt = """You are a smart AI assistant that helps users based on the provided context.

Your tasks:
1. Understand exactly what the user is asking.
2. Use ONLY the provided context to answer when possible.
3. If the context is not enough, say so clearly.
4. Be precise, clear, and helpful.
5. If the user is asking for an action (like create Excel, generate PDF, send email, etc.), 
   clearly identify the requested action and what information is needed.

Always reply in the same language the user used."""

    user_prompt = f"""Context:
{context}

User Query:
{query}

Please analyze the query and respond accordingly."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def reason(query: str, top_k: int = TOP_K) -> Dict:
    """
    Main reasoning function.
    
    1. Retrieves relevant chunks
    2. Builds context
    3. Calls the LLM
    4. Returns structured response
    """
    logger.info(f"Processing query: {query}")

    # 1. Load retriever (cached)
    model, collection = load_retriever()

    # 2. Retrieve relevant chunks
    results = retrieve(query, model=model, collection=collection, top_k=top_k)
    logger.info(f"Retrieved {len(results)} chunks")

    # 3. Build context + prompt
    context = build_context(results)
    messages = build_prompt(query, context)

    # 4. Call LLM
    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": results,
        "context_used": context
    }


def print_response(result: Dict) -> None:
    """Pretty print the final response."""
    print("\n" + "=" * 80)
    print(f"Query : {result['query']}")
    print("=" * 80)
    print("\nAnswer:\n")
    print(result["answer"])
    print("\n" + "-" * 80)
    print(f"Used {len(result['retrieved_chunks'])} chunks from the knowledge base.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_queries = [
            "What is the attention mechanism?",
            "Explain how RAG works according to the paper.",
            "What are the main contributions of the GPT-3 paper?",
        ]

        for q in test_queries:
            result = reason(q)
            print_response(result)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")