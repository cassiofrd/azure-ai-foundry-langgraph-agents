from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    source_path: str
    file_type: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EnrichedDocument:
    document_id: str
    source_path: str
    file_type: str
    content: str
    domain: str
    document_type: str
    entity_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    location_ids: tuple[str, ...] = ()
    supplier_ids: tuple[str, ...] = ()
    is_global: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_path: str
    file_type: str
    content: str
    domain: str
    document_type: str
    entity_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    location_ids: tuple[str, ...] = ()
    supplier_ids: tuple[str, ...] = ()
    is_global: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
