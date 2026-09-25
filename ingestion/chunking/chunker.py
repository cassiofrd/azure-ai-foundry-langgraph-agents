from __future__ import annotations

import re

from ingestion.models import DocumentChunk, EnrichedDocument


TABULAR_RECORD_TYPES = {"csv_row", "xlsx_row"}


def _split_pdf_text(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    # Some extracted PDFs have one field/value per line and therefore no
    # blank-line paragraphs. Fall back to line groups in that case.
    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    chunks: list[str] = []
    current = ""

    for part in paragraphs:
        candidate = part if not current else f"{current}\n{part}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

            overlap = current[-overlap_chars:].strip()
            current = f"{overlap}\n{part}".strip() if overlap else part
        else:
            # Very long single segment: use a character window fallback.
            start = 0
            step = max(1, max_chars - overlap_chars)
            while start < len(part):
                piece = part[start:start + max_chars].strip()
                if piece:
                    chunks.append(piece)
                start += step
            current = ""

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_document(
    document: EnrichedDocument,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[DocumentChunk]:
    record_type = document.metadata.get("record_type", "")

    # CSV/XLSX rows are already small, meaningful atomic records.
    if record_type in TABULAR_RECORD_TYPES:
        texts = [document.content.strip()] if document.content.strip() else []
    else:
        texts = _split_pdf_text(
            document.content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    chunks: list[DocumentChunk] = []

    for index, text in enumerate(texts, start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}#chunk={index}",
                document_id=document.document_id,
                source_path=document.source_path,
                file_type=document.file_type,
                content=text,
                domain=document.domain,
                document_type=document.document_type,
                entity_ids=document.entity_ids,
                aliases=document.aliases,
                location_ids=document.location_ids,
                supplier_ids=document.supplier_ids,
                is_global=document.is_global,
                metadata={
                    **document.metadata,
                    "chunk_number": str(index),
                },
            )
        )

    return chunks


def chunk_documents(
    documents: list[EnrichedDocument],
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return chunks
