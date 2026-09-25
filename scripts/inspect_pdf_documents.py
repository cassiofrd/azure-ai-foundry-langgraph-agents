from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
from ingestion.parsers.pdf_parser import parse_pdf_bytes


def _preview(text: str, max_lines: int = 12) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview_lines = lines[:max_lines]
    preview = "\n".join(f"    {line}" for line in preview_lines)

    if len(lines) > max_lines:
        preview += f"\n    ... {len(lines) - max_lines} more line(s)"

    return preview


def main():
    pdf_paths = sorted(
        path for path in list_source_documents()
        if path.lower().endswith(".pdf")
    )

    if not pdf_paths:
        print("No PDF source documents found.")
        return

    total_pages = 0

    for path in pdf_paths:
        data = download_source_document(path)
        documents = parse_pdf_bytes(data=data, source_path=path)
        total_pages += len(documents)

        print(f"\n{path}")
        print(f"Parsed pages with text: {len(documents)}")

        for document in documents[:2]:
            print(f"\n  ID: {document.document_id}")
            print(f"  Metadata: {document.metadata}")
            print("  Content preview:")
            print(_preview(document.content))

        if len(documents) > 2:
            print(f"\n  ... {len(documents) - 2} more page(s)")

    print(f"\nTotal parsed PDF pages with text: {total_pages}")


if __name__ == "__main__":
    main()
