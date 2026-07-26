from __future__ import annotations

from dataclasses import dataclass

from ai07_chunker import Chunk, build_chunks, parse_markdown_blocks
from ai08_vector_retrieval import (
    EmbeddedChunk,
    RetrievalResult,
    SAMPLE_MARKDOWN,
    embed_chunks,
    retrieve_top_k,
    tokenize,
)


@dataclass
class RetrievalComparison:
    keyword_results: list[RetrievalResult]
    vector_results: list[RetrievalResult]


def keyword_score(query: str, chunk: Chunk) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    chunk_tokens = set(tokenize(chunk.text))
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def keyword_retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    k: int = 3,
) -> list[RetrievalResult]:
    results = [
        RetrievalResult(chunk=chunk, score=keyword_score(query, chunk))
        for chunk in chunks
    ]
    results.sort(
        key=lambda item: (
            item.score,
            -int(item.chunk.metadata.get("chunk_index", 0)),
        ),
        reverse=True,
    )
    return results[:k]


def compare_retrieval(
    query: str,
    chunks: list[Chunk],
    embedded_chunks: list[EmbeddedChunk],
    k: int = 3,
) -> RetrievalComparison:
    return RetrievalComparison(
        keyword_results=keyword_retrieve_top_k(query, chunks, k=k),
        vector_results=retrieve_top_k(query, embedded_chunks, k=k),
    )


def format_heading_path(chunk: Chunk) -> str:
    heading_path = chunk.metadata.get("heading_path", [])
    if not heading_path:
        return "(no heading)"
    return " > ".join(str(item) for item in heading_path)


def format_context_item(index: int, result: RetrievalResult) -> str:
    metadata = result.chunk.metadata
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index", "unknown")
    heading = format_heading_path(result.chunk)
    return (
        f"[{index}] source={source}; heading={heading}; "
        f"chunk_index={chunk_index}; score={result.score:.4f}\n"
        f"{result.chunk.text}"
    )


def build_rag_prompt(query: str, results: list[RetrievalResult]) -> str:
    context = "\n\n".join(
        format_context_item(index, result)
        for index, result in enumerate(results, start=1)
    )
    return f"""You are answering based only on the provided context.
If the context is insufficient, say what is missing instead of guessing.

Question:
{query}

Context:
{context}

Answer:
"""


def print_results(title: str, results: list[RetrievalResult]) -> None:
    print(title)
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. score={result.score:.4f} "
            f"heading={format_heading_path(result.chunk)}"
        )
    print()


def demo() -> None:
    blocks = parse_markdown_blocks(SAMPLE_MARKDOWN)
    chunks = build_chunks(blocks, source="SearchEngine sample", max_chars=260)
    embedded_chunks = embed_chunks(chunks)

    query = "How does the cache improve search performance?"
    comparison = compare_retrieval(query, chunks, embedded_chunks, k=2)

    assert comparison.keyword_results
    assert comparison.vector_results
    assert "Cache Module" in comparison.keyword_results[0].chunk.metadata["heading_path"]
    assert "Cache Module" in comparison.vector_results[0].chunk.metadata["heading_path"]

    print_results("Keyword TopK", comparison.keyword_results)
    print_results("Vector TopK", comparison.vector_results)
    print("RAG Prompt")
    print(build_rag_prompt(query, comparison.vector_results))


if __name__ == "__main__":
    demo()
