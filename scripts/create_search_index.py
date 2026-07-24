from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from shared.settings import AppSettings, load_settings


HNSW_ALGORITHM_NAME = "supply-chain-hnsw"
VECTOR_PROFILE_NAME = "supply-chain-vector-profile"


def validate_search_settings(
    settings: AppSettings,
) -> None:
    if not settings.azure_search_endpoint:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT is required."
        )

    if not settings.azure_search_index_name:
        raise ValueError(
            "AZURE_SEARCH_INDEX_NAME is required."
        )

    if not settings.azure_search_admin_key:
        raise ValueError(
            "AZURE_SEARCH_ADMIN_KEY is required."
        )

    if not settings.azure_search_vector_field:
        raise ValueError(
            "AZURE_SEARCH_VECTOR_FIELD is required."
        )

    if settings.azure_search_vector_dimensions < 1:
        raise ValueError(
            "AZURE_SEARCH_VECTOR_DIMENSIONS must be "
            "greater than zero."
        )


def build_index(
    settings: AppSettings,
) -> SearchIndex:
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="agent",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="doc_type",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="entity_type",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="entity_id",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),
        SearchableField(
            name="source",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name=settings.azure_search_vector_field,
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=(
                settings.azure_search_vector_dimensions
            ),
            vector_search_profile_name=(
                VECTOR_PROFILE_NAME
            ),
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_ALGORITHM_NAME,
                parameters=HnswParameters(
                    metric="cosine",
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=(
                    HNSW_ALGORITHM_NAME
                ),
            )
        ],
    )

    return SearchIndex(
        name=settings.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
    )


def create_index_client(
    settings: AppSettings,
) -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(
            settings.azure_search_admin_key
        ),
    )


def delete_existing_index(
    *,
    client: SearchIndexClient,
    index_name: str,
) -> None:
    try:
        client.delete_index(index_name)

    except ResourceNotFoundError:
        print(
            "No existing Azure AI Search index was found. "
            "A new index will be created."
        )
        return

    print(
        "Existing Azure AI Search index deleted "
        f"successfully: {index_name}"
    )


def main() -> None:
    settings = load_settings()

    validate_search_settings(settings)

    client = create_index_client(settings)

    delete_existing_index(
        client=client,
        index_name=settings.azure_search_index_name,
    )

    index = build_index(settings)
    result = client.create_index(index)

    print(
        "Azure AI Search index created successfully: "
        f"{result.name}"
    )
    print(
        "Vector field: "
        f"{settings.azure_search_vector_field}"
    )
    print(
        "Vector dimensions: "
        f"{settings.azure_search_vector_dimensions}"
    )
    print(
        "Vector profile: "
        f"{VECTOR_PROFILE_NAME}"
    )


if __name__ == "__main__":
    main()