from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import sqrt
import re

from ai07_chunker import Chunk, build_chunks, parse_markdown_blocks


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def fake_embed_text(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dim
        vector[index] += 1.0
    return normalize(vector)


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(chunk=chunk, vector=fake_embed_text(chunk.text))
        for chunk in chunks
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def retrieve_top_k(
    query: str,
    embedded_chunks: list[EmbeddedChunk],
    k: int = 3,
) -> list[RetrievalResult]:
    query_vector = fake_embed_text(query)
    results = [
        RetrievalResult(
            chunk=item.chunk,
            score=cosine_similarity(query_vector, item.vector),
        )
        for item in embedded_chunks
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:k]


SAMPLE_MARKDOWN = """# SearchEngine

## Network Module

The online stage uses socket, epoll, Reactor, EventLoop and TcpConnection.
Expensive search tasks run in worker threads.

## Cache Module

Redis caches keyword suggestion results.
The local LRUCache caches web search results.
When Redis is unavailable, the service falls back to the normal search path.

## Index Module

The offline stage parses web pages, removes duplicates with Simhash,
and builds inverted index and TF-IDF weights.
"""


def demo() -> None:
    blocks = parse_markdown_blocks(SAMPLE_MARKDOWN)
    chunks = build_chunks(blocks, source="SearchEngine sample", max_chars=260)
    embedded_chunks = embed_chunks(chunks)

    query = "How does the cache improve search performance?"
    results = retrieve_top_k(query, embedded_chunks, k=2)

    assert results
    assert "Cache Module" in results[0].chunk.metadata["heading_path"]

    for result in results:
        print(f"score={result.score:.4f}")
        print(result.chunk.metadata)
        print(result.chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    demo()
