from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

from ai07_chunker import Chunk, build_chunks, parse_markdown_blocks
from ai08_vector_retrieval import (
    EmbeddedChunk,
    RetrievalResult,
    SAMPLE_MARKDOWN,
    embed_chunks,
    retrieve_top_k,
)
from ai10_ai11_rag_prompt import keyword_retrieve_top_k


RetrievalMode = Literal["keyword", "vector", "hybrid"]
ToolStatus = Literal["ok", "empty", "timeout", "error", "degraded"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    heading_path: str
    score: float
    retrieval_mode: str


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    tool_name: str
    query: str
    results: list[RetrievedChunk]
    latency_ms: int
    cache_hit: bool = False
    degraded: bool = False
    error_code: str | None = None


SEARCH_PROJECT_DOCS_SCHEMA = {
    "name": "search_project_docs",
    "args": {
        "query": {
            "type": "string",
            "required": True,
            "max_length": 200,
        },
        "top_k": {
            "type": "integer",
            "required": False,
            "default": 3,
            "min": 1,
            "max": 10,
        },
        "mode": {
            "type": "enum",
            "required": False,
            "default": "hybrid",
            "values": ["keyword", "vector", "hybrid"],
        },
    },
}


class RetrievalCache:
    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, ToolResult] = OrderedDict()

    def get(self, key: str) -> ToolResult | None:
        result = self._items.get(key)
        if result is None:
            return None

        self._items.move_to_end(key)
        return replace(result, cache_hit=True, latency_ms=0)

    def set(self, key: str, result: ToolResult) -> None:
        self._items[key] = replace(result, cache_hit=False)
        self._items.move_to_end(key)
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)


class KeywordRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [
            result
            for result in keyword_retrieve_top_k(query, self.chunks, k=top_k)
            if result.score > 0
        ]


class VectorRetriever:
    def __init__(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self.embedded_chunks = embedded_chunks

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if "__simulate_vector_timeout__" in query:
            raise TimeoutError("simulated vector retrieval timeout")
        return [
            result
            for result in retrieve_top_k(query, self.embedded_chunks, k=top_k)
            if result.score > 0
        ]


class SearchProjectDocsTool:
    def __init__(
        self,
        cache: RetrievalCache,
        keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
    ) -> None:
        self.cache = cache
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever

    def execute(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
    ) -> ToolResult:
        start = perf_counter()
        cache_key = self._build_cache_key(query, top_k, mode)

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            results = self._retrieve(query, top_k, mode)
            status: ToolStatus = "ok" if results else "empty"
            tool_result = ToolResult(
                status=status,
                tool_name="search_project_docs",
                query=query,
                results=results,
                latency_ms=self._elapsed_ms(start),
            )
            self.cache.set(cache_key, tool_result)
            return tool_result
        except TimeoutError:
            return self._fallback_to_keyword(query, top_k, mode, start)

    def _retrieve(
        self,
        query: str,
        top_k: int,
        mode: RetrievalMode,
    ) -> list[RetrievedChunk]:
        if mode == "keyword":
            return self._wrap_results(
                self.keyword_retriever.retrieve(query, top_k),
                retrieval_mode="keyword",
            )

        if mode == "vector":
            return self._wrap_results(
                self.vector_retriever.retrieve(query, top_k),
                retrieval_mode="vector",
            )

        keyword_results = self._wrap_results(
            self.keyword_retriever.retrieve(query, top_k * 2),
            retrieval_mode="keyword",
        )
        vector_results = self._wrap_results(
            self.vector_retriever.retrieve(query, top_k * 2),
            retrieval_mode="vector",
        )
        return self._merge_results(keyword_results, vector_results, top_k)

    def _fallback_to_keyword(
        self,
        query: str,
        top_k: int,
        mode: RetrievalMode,
        start: float,
    ) -> ToolResult:
        if mode not in {"vector", "hybrid"}:
            return ToolResult(
                status="timeout",
                tool_name="search_project_docs",
                query=query,
                results=[],
                latency_ms=self._elapsed_ms(start),
                error_code="retrieval_timeout",
            )

        results = self._wrap_results(
            self.keyword_retriever.retrieve(query, top_k),
            retrieval_mode="keyword",
        )
        return ToolResult(
            status="degraded" if results else "empty",
            tool_name="search_project_docs",
            query=query,
            results=results,
            latency_ms=self._elapsed_ms(start),
            degraded=True,
            error_code="vector_timeout",
        )

    def _merge_results(
        self,
        keyword_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for result in keyword_results + vector_results:
            existing = merged.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                merged[result.chunk_id] = result

        return sorted(
            merged.values(),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    def _wrap_results(
        self,
        results: list[RetrievalResult],
        retrieval_mode: str,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=str(result.chunk.metadata.get("chunk_index", "")),
                content=result.chunk.text,
                source=str(result.chunk.metadata.get("source", "unknown")),
                heading_path=self._format_heading_path(result.chunk),
                score=result.score,
                retrieval_mode=retrieval_mode,
            )
            for result in results
        ]

    def _build_cache_key(self, query: str, top_k: int, mode: RetrievalMode) -> str:
        normalized_query = " ".join(query.lower().split())
        return f"retrieval:{mode}:{top_k}:{normalized_query}"

    def _format_heading_path(self, chunk: Chunk) -> str:
        heading_path = chunk.metadata.get("heading_path", [])
        if not heading_path:
            return "(no heading)"
        return " > ".join(str(item) for item in heading_path)

    def _elapsed_ms(self, start: float) -> int:
        return int((perf_counter() - start) * 1000)


class ToolDispatcher:
    def __init__(self, tools: dict[str, SearchProjectDocsTool]) -> None:
        self.tools = tools

    def dispatch(self, call: ToolCall) -> ToolResult:
        if call.name not in self.tools:
            return ToolResult(
                status="error",
                tool_name=call.name,
                query="",
                results=[],
                latency_ms=0,
                error_code="unknown_tool",
            )

        try:
            args = self._validate_and_normalize(call.arguments)
        except ValueError as exc:
            return ToolResult(
                status="error",
                tool_name=call.name,
                query=str(call.arguments.get("query", "")),
                results=[],
                latency_ms=0,
                error_code=str(exc),
            )

        return self.tools[call.name].execute(**args)

    def _validate_and_normalize(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query_required")
        if len(query) > SEARCH_PROJECT_DOCS_SCHEMA["args"]["query"]["max_length"]:
            raise ValueError("query_too_long")

        top_k = int(args.get("top_k", SEARCH_PROJECT_DOCS_SCHEMA["args"]["top_k"]["default"]))
        min_top_k = SEARCH_PROJECT_DOCS_SCHEMA["args"]["top_k"]["min"]
        max_top_k = SEARCH_PROJECT_DOCS_SCHEMA["args"]["top_k"]["max"]
        if top_k < min_top_k or top_k > max_top_k:
            raise ValueError("top_k_out_of_range")

        mode = args.get("mode", SEARCH_PROJECT_DOCS_SCHEMA["args"]["mode"]["default"])
        if mode not in SEARCH_PROJECT_DOCS_SCHEMA["args"]["mode"]["values"]:
            raise ValueError("invalid_retrieval_mode")

        return {
            "query": query,
            "top_k": top_k,
            "mode": mode,
        }


class AgentRunner:
    def __init__(self, dispatcher: ToolDispatcher, max_tool_calls: int = 3) -> None:
        self.dispatcher = dispatcher
        self.max_tool_calls = max_tool_calls

    def run(self, user_query: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "Use project document tools when project facts are needed.",
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        tool_calls_used = 0
        while tool_calls_used < self.max_tool_calls:
            model_output = self._mock_llm_plan(messages)
            if "tool_call" not in model_output:
                return str(model_output["answer"])

            raw_tool_call = model_output["tool_call"]
            tool_call = ToolCall(
                name=raw_tool_call["name"],
                arguments=raw_tool_call["arguments"],
            )
            tool_result = self.dispatcher.dispatch(tool_call)
            tool_calls_used += 1

            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "content": self._format_observation(tool_result),
                }
            )
            return self._mock_llm_answer(tool_result)

        return "Tool call limit reached before a final answer was produced."

    def _mock_llm_plan(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_query = str(messages[-1]["content"])
        return {
            "tool_call": {
                "name": "search_project_docs",
                "arguments": {
                    "query": user_query,
                    "top_k": 3,
                    "mode": "hybrid",
                },
            }
        }

    def _mock_llm_answer(self, tool_result: ToolResult) -> str:
        if tool_result.status == "empty":
            return "No relevant project documents were retrieved."

        if tool_result.status in {"timeout", "error"}:
            return f"Retrieval failed: {tool_result.error_code}"

        lines = [
            (
                f"Retrieved {len(tool_result.results)} chunks "
                f"(status={tool_result.status}, cache_hit={tool_result.cache_hit}, "
                f"degraded={tool_result.degraded})."
            )
        ]
        for item in tool_result.results:
            lines.append(
                f"- {item.heading_path} [{item.retrieval_mode}] "
                f"score={item.score:.4f}: {item.content[:120]}"
            )
        return "\n".join(lines)

    def _format_observation(self, tool_result: ToolResult) -> dict[str, Any]:
        return {
            "status": tool_result.status,
            "query": tool_result.query,
            "result_count": len(tool_result.results),
            "cache_hit": tool_result.cache_hit,
            "degraded": tool_result.degraded,
            "error_code": tool_result.error_code,
            "results": [
                {
                    "content": item.content,
                    "source": item.source,
                    "heading_path": item.heading_path,
                    "score": item.score,
                    "retrieval_mode": item.retrieval_mode,
                }
                for item in tool_result.results
            ],
        }


def build_demo_agent() -> AgentRunner:
    blocks = parse_markdown_blocks(SAMPLE_MARKDOWN)
    chunks = build_chunks(blocks, source="SearchEngine sample", max_chars=260)
    embedded_chunks = embed_chunks(chunks)

    cache = RetrievalCache(capacity=8)
    search_tool = SearchProjectDocsTool(
        cache=cache,
        keyword_retriever=KeywordRetriever(chunks),
        vector_retriever=VectorRetriever(embedded_chunks),
    )
    dispatcher = ToolDispatcher(
        tools={
            "search_project_docs": search_tool,
        }
    )
    return AgentRunner(dispatcher=dispatcher)


def demo() -> None:
    agent = build_demo_agent()

    query = "How does SearchEngine cache repeated search queries?"
    first_answer = agent.run(query)
    second_answer = agent.run(query)

    print("First request")
    print(first_answer)
    print()
    print("Second request")
    print(second_answer)

    assert "Cache Module" in first_answer
    assert "cache_hit=False" in first_answer
    assert "cache_hit=True" in second_answer

    degraded_answer = agent.run(
        "__simulate_vector_timeout__ cache repeated search queries"
    )
    print()
    print("Degraded request")
    print(degraded_answer)
    assert "degraded=True" in degraded_answer


if __name__ == "__main__":
    demo()
