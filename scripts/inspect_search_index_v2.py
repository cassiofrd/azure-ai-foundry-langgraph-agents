from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from scripts.create_search_index_v2 import INDEX_NAME
from shared.settings import load_settings


def main():
    settings = load_settings()

    client = SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(
            settings.azure_search_admin_key
        ),
    )

    index = client.get_index(INDEX_NAME)

    print(f"Index: {index.name}")
    print(f"Fields: {len(index.fields)}")

    for field in index.fields:
        dimensions = getattr(
            field,
            "vector_search_dimensions",
            None,
        )

        print(
            f"- {field.name}: "
            f"type={field.type}, "
            f"searchable={field.searchable}, "
            f"filterable={field.filterable}, "
            f"facetable={field.facetable}"
            + (
                f", dimensions={dimensions}"
                if dimensions is not None
                else ""
            )
        )


if __name__ == "__main__":
    main()
