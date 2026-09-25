from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
from ingestion.parsers.xlsx_parser import parse_xlsx_bytes


def main():
    xlsx_paths = sorted(
        path
        for path in list_source_documents()
        if path.lower().endswith(".xlsx")
    )

    if not xlsx_paths:
        print("No XLSX source documents found.")
        return

    total_records = 0

    for path in xlsx_paths:
        data = download_source_document(path)
        documents = parse_xlsx_bytes(
            data=data,
            source_path=path,
        )

        total_records += len(documents)

        print(f"\n{path}")
        print(f"Parsed records: {len(documents)}")

        for document in documents[:3]:
            print(f"\n  ID: {document.document_id}")
            print(f"  Metadata: {document.metadata}")
            print("  Content:")

            for line in document.content.splitlines():
                print(f"    {line}")

        if len(documents) > 3:
            print(
                f"\n  ... {len(documents) - 3} more record(s)"
            )

    print(f"\nTotal parsed XLSX records: {total_records}")


if __name__ == "__main__":
    main()
