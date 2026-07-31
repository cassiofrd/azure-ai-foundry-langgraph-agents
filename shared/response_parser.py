from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    title: str
    source: str
    entity_id: str


_INLINE_EVIDENCE_PATTERN = re.compile(
    r"(?:-\s*)?Title:\s*(?P<title>.*?),\s*"
    r"Source:\s*(?P<source>.*?),\s*"
    r"(?:Entity_id|entity_id):\s*(?P<entity_id>[^\n]+)",
    re.IGNORECASE,
)


def extract_evidence(answer: str) -> tuple[EvidenceItem, ...]:
    """Extract and deduplicate evidence metadata from an agent answer."""
    normalized = answer.strip()
    if not normalized:
        return ()

    items: list[EvidenceItem] = []

    for match in _INLINE_EVIDENCE_PATTERN.finditer(normalized):
        items.append(
            EvidenceItem(
                title=match.group("title").strip(),
                source=match.group("source").strip(),
                entity_id=match.group("entity_id").strip(),
            )
        )

    current: dict[str, str] = {}
    for raw_line in normalized.splitlines():
        line = raw_line.strip().lstrip("- ").strip()
        lowered = line.lower()
        if lowered.startswith("title:"):
            if current:
                _append_if_complete(items, current)
                current = {}
            current["title"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("source:"):
            current["source"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("entity_id:"):
            current["entity_id"] = line.split(":", 1)[1].strip()
            _append_if_complete(items, current)
            current = {}

    _append_if_complete(items, current)

    unique: dict[tuple[str, str, str], EvidenceItem] = {}
    for item in items:
        key = (
            item.title.casefold(),
            item.source.casefold(),
            item.entity_id.casefold(),
        )
        unique.setdefault(key, item)
    return tuple(unique.values())


def _append_if_complete(
    items: list[EvidenceItem],
    values: dict[str, str],
) -> None:
    title = values.get("title", "").strip()
    source = values.get("source", "").strip()
    entity_id = values.get("entity_id", "").strip()
    if title and source and entity_id:
        items.append(
            EvidenceItem(
                title=title,
                source=source,
                entity_id=entity_id,
            )
        )
