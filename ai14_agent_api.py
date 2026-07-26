from __future__ import annotations

from time import perf_counter
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from ai12_ai13_agent_tool import RetrievedChunk, ToolCall, ToolResult, build_demo_agent


RetrievalMode = Literal["keyword", "vector", "hybrid"]

app = FastAPI(title="SearchEngine-AI Agent API")
agent = build_demo_agent()


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=10)
    mode: RetrievalMode = "hybrid"

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/query")
def agent_query(request: AgentQueryRequest) -> dict[str, object]:
    start = perf_counter()
    tool_result = agent.dispatcher.dispatch(
        ToolCall(
            name="search_project_docs",
            arguments={
                "query": request.query,
                "top_k": request.top_k,
                "mode": request.mode,
            },
        )
    )

    return {
        "answer": _build_answer(tool_result),
        "results": [_serialize_result(item) for item in tool_result.results],
        "cache_hit": tool_result.cache_hit,
        "degraded": tool_result.degraded,
        "status": tool_result.status,
        "error_code": tool_result.error_code,
        "latency_ms": int((perf_counter() - start) * 1000),
        "retrieval_latency_ms": tool_result.latency_ms,
    }


def _build_answer(tool_result: ToolResult) -> str:
    if tool_result.status == "empty":
        return "No relevant project documents were retrieved."

    if tool_result.status in {"timeout", "error"}:
        return f"Retrieval failed: {tool_result.error_code}"

    return (
        f"Retrieved {len(tool_result.results)} project document chunks "
        f"with status={tool_result.status}, cache_hit={tool_result.cache_hit}, "
        f"degraded={tool_result.degraded}."
    )


def _serialize_result(item: RetrievedChunk) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "content": item.content,
        "source": item.source,
        "heading_path": item.heading_path,
        "score": item.score,
        "retrieval_mode": item.retrieval_mode,
    }
