from __future__ import annotations

import csv
import io
from pathlib import PurePosixPath

from ingestion.models import ParsedDocument


def _detect_dialect(text: str) -> csv.Dialect:
    sample = text[:4096]

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.get_dialect("excel")


def parse_csv_bytes(
    data: bytes,
    source_path: str,
) -> list[ParsedDocument]:
    text = data.decode("utf-8-sig")
    dialect = _detect_dialect(text)

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    if not reader.fieldnames:
        raise ValueError(
            f"CSV file has no header row: {source_path}"
        )

    source_file = PurePosixPath(source_path).name
    domain = source_path.split("/", 1)[0] if "/" in source_path else ""

    documents: list[ParsedDocument] = []

    for row_number, row in enumerate(reader, start=2):
        cleaned_row = {
            (key or "").strip(): (value or "").strip()
            for key, value in row.items()
        }

        content = "\n".join(
            f"{key}: {value}"
            for key, value in cleaned_row.items()
            if key
        )

        documents.append(
            ParsedDocument(
                document_id=f"{source_path}#row={row_number}",
                source_path=source_path,
                file_type="csv",
                content=content,
                metadata={
                    "domain": domain,
                    "source_file": source_file,
                    "record_type": "csv_row",
                    "row_number": str(row_number),
                },
            )
        )

    return documents
