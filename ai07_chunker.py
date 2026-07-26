from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re


@dataclass
class Block:
    kind: str
    text: str
    heading_path: list[str]
    start_char: int
    end_char: int


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    heading_path: list[str] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0

    lines = text.splitlines(keepends=True)
    char_pos = 0
    i = 0

    def flush_paragraph(end_pos: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if not paragraph_lines:
            return

        paragraph = "".join(paragraph_lines).strip()
        if paragraph:
            blocks.append(
                Block(
                    kind="paragraph",
                    text=paragraph,
                    heading_path=heading_path.copy(),
                    start_char=paragraph_start,
                    end_char=end_pos,
                )
            )
        paragraph_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_start = char_pos
        line_end = char_pos + len(line)

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph(line_start)
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            heading_path = heading_path[: level - 1] + [title]
            blocks.append(
                Block(
                    kind="heading",
                    text=stripped,
                    heading_path=heading_path.copy(),
                    start_char=line_start,
                    end_char=line_end,
                )
            )
            char_pos = line_end
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph(line_start)
            code_start = line_start
            code_lines = [line]
            char_pos = line_end
            i += 1

            while i < len(lines):
                code_line = lines[i]
                code_lines.append(code_line)
                char_pos += len(code_line)
                i += 1
                if code_line.strip().startswith("```"):
                    break

            blocks.append(
                Block(
                    kind="code",
                    text="".join(code_lines).strip(),
                    heading_path=heading_path.copy(),
                    start_char=code_start,
                    end_char=char_pos,
                )
            )
            continue

        if stripped == "":
            flush_paragraph(line_start)
            char_pos = line_end
            i += 1
            continue

        if not paragraph_lines:
            paragraph_start = line_start
        paragraph_lines.append(line)
        char_pos = line_end
        i += 1

    flush_paragraph(char_pos)
    return blocks


def build_chunks(blocks: list[Block], source: str, max_chars: int = 800) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_blocks: list[Block] = []

    def joined_length(items: list[Block]) -> int:
        if not items:
            return 0
        return sum(len(block.text) for block in items) + 2 * (len(items) - 1)

    def emit(items: list[Block]) -> None:
        if not items:
            return

        chunk_text = "\n\n".join(block.text for block in items)
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    "source": source,
                    "heading_path": items[0].heading_path,
                    "chunk_index": len(chunks),
                    "start_char": items[0].start_char,
                    "end_char": items[-1].end_char,
                    "block_kinds": [block.kind for block in items],
                    "char_count": len(chunk_text),
                },
            )
        )

    for block in blocks:
        if block.kind == "heading" and current_blocks:
            emit(current_blocks)
            current_blocks = []

        candidate = current_blocks + [block]
        if joined_length(candidate) <= max_chars:
            current_blocks = candidate
            continue

        emit(current_blocks)
        current_blocks = [block]

    emit(current_blocks)
    return chunks


SAMPLE_MARKDOWN = """# SearchEngine

The project has an offline index phase and an online query phase.

## Network Module

The online stage uses socket, epoll, Reactor, EventLoop and TcpConnection.
Expensive search tasks run in worker threads, and responses are sent back in the IO thread.

```cpp
pool.addTask([conn, query] {
    auto result = search(query);
    conn->sendInLoop(result);
});
```

## Cache Module

Redis caches keyword suggestion results.
The local LRUCache caches web search results.
When Redis is unavailable, the service falls back to the normal search path.
"""


def demo() -> None:
    blocks = parse_markdown_blocks(SAMPLE_MARKDOWN)
    chunks = build_chunks(blocks, source="SearchEngine README sample", max_chars=320)

    assert any(block.kind == "code" for block in blocks)
    assert any("Network Module" in chunk.metadata["heading_path"] for chunk in chunks)
    assert any("Cache Module" in chunk.metadata["heading_path"] for chunk in chunks)
    assert all(chunk.metadata["char_count"] <= 320 for chunk in chunks)
    assert all(
        not ("## Cache Module" in chunk.text and "Network Module" in chunk.metadata["heading_path"])
        for chunk in chunks
    )

    for chunk in chunks:
        print(json.dumps(chunk.metadata, ensure_ascii=False))
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    demo()
