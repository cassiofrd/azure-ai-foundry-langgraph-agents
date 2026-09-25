from ingestion.blob_reader import list_source_documents


def main():
    documents = list_source_documents()

    print(f"Found {len(documents)} source documents:\n")

    for document in documents:
        print(f"- {document}")


if __name__ == "__main__":
    main()