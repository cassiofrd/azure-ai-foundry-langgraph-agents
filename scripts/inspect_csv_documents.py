from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
from ingestion.parsers.csv_parser import parse_csv_bytes


def main():
    csv_paths = sorted(
        path
        for path in list_source_documents()
        if path.lower().endswith(".csv")
    )

    if not csv_paths:
        print("No CSV source documents found.")
        return

    total_records = 0

    for path in csv_paths:
        data = download_source_document(path)
        documents = parse_csv_bytes(
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
            print(f"\n  ... {len(documents) - 3} more record(s)")

    print(f"\nTotal parsed CSV records: {total_records}")


if __name__ == "__main__":
    main()
