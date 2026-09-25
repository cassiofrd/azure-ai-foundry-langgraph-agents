from __future__ import annotations

from pathlib import PurePosixPath

import pymupdf

from ingestion.models import ParsedDocument


def parse_pdf_bytes(
    data: bytes,
    source_path: str,
) -> list[ParsedDocument]:
    pdf = pymupdf.open(stream=data, filetype="pdf")

    source_file = PurePosixPath(source_path).name
    domain = source_path.split("/", 1)[0] if "/" in source_path else ""

    documents: list[ParsedDocument] = []

    try:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()

            if not text:
                continue

            page_number = page_index + 1

            documents.append(
                ParsedDocument(
                    document_id=f"{source_path}#page={page_number}",
                    source_path=source_path,
                    file_type="pdf",
                    content=text,
                    metadata={
                        "domain": domain,
                        "source_file": source_file,
                        "record_type": "pdf_page",
                        "page_number": str(page_number),
                    },
                )
            )
    finally:
        pdf.close()

    return documents
