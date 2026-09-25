from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from shared.settings import load_settings


def get_container_client():
    settings = load_settings()

    account_url = (
        f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
    )

    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    return blob_service_client.get_container_client(
        settings.azure_storage_container_name
    )


def list_source_documents() -> list[str]:
    container_client = get_container_client()

    return [
        blob.name
        for blob in container_client.list_blobs()
        if not blob.name.endswith("/")
    ]


def download_source_document(blob_name: str) -> bytes:
    container_client = get_container_client()
    blob_client = container_client.get_blob_client(blob_name)

    return blob_client.download_blob().readall()
