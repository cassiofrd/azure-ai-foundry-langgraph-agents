from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from ingestion.pipeline import INDEX_NAME
from shared.settings import load_settings


def main():
    settings = load_settings()
    client = SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(settings.azure_search_admin_key),
    )

    print(f"Index: {INDEX_NAME}")
    print(f"Document count: {client.get_document_count()}\n")

    results = client.search(
        search_text="*",
        filter="entity_ids/any(e: e eq 'M10')",
        select=[
            "chunk_id", "domain", "document_type", "entity_ids",
            "aliases", "location_ids", "is_global",
        ],
        top=5,
    )

    print("Sample M10 documents:")
    for result in results:
        print(f"- {result['chunk_id']}")
        print(f"  domain: {result.get('domain')}")
        print(f"  document_type: {result.get('document_type')}")
        print(f"  entity_ids: {result.get('entity_ids')}")
        print(f"  aliases: {result.get('aliases')}")
        print(f"  location_ids: {result.get('location_ids')}")
        print(f"  is_global: {result.get('is_global')}\n")


if __name__ == "__main__":
    main()
