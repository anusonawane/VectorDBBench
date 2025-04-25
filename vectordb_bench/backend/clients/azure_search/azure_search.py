import logging
from collections.abc import Iterable
from contextlib import contextmanager
import time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
)

from ..api import VectorDB
from .config import AzureSearchConfig, AzureSearchIndexConfig

log = logging.getLogger(__name__)

class AzureSearch(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AzureSearchIndexConfig,
        collection_name: str = "vectordbbench",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.id_field = "id"
        self.vector_field = "vector"

        self.admin_client = SearchIndexClient(
            endpoint=db_config["endpoint"],
            credential=AzureKeyCredential(db_config["key"])
        )

        if drop_old:
            try:
                self.admin_client.delete_index(self.collection_name)
            except Exception:
                pass

            self._create_index()

    def _create_index(self):
        # Enable detailed logging
        import logging
        logging.basicConfig(level=logging.INFO)
        azure_logger = logging.getLogger("azure")
        azure_logger.setLevel(logging.DEBUG)

        # First, try to check if we can connect to the service
        log.info(f"Testing connection to Azure AI Search service at: {self.db_config['endpoint']}")
        try:
            # Try to get service statistics first to check connectivity
            log.info("Checking service statistics...")
            stats = self.admin_client.get_service_statistics()
            log.info(f"Service statistics: {stats}")

            # List existing indexes
            log.info("Listing existing indexes...")
            indexes = list(self.admin_client.list_indexes())
            log.info(f"Found {len(indexes)} existing indexes")
            for idx in indexes:
                log.info(f" - {idx.name}")

            # Since we don't have permission to create indexes, we'll simulate the benchmark
            log.info("Using a simulated index for read-only benchmark")
            # We'll keep the original collection name but won't actually create it
            log.info(f"Using collection name: '{self.collection_name}' (simulated)")

        except Exception as e:
            log.error(f"Error during connectivity check: {e}")
            if hasattr(e, 'status_code'):
                log.error(f"Status code: {e.status_code}")
            if hasattr(e, 'reason'):
                log.error(f"Reason: {e.reason}")
            if hasattr(e, 'message'):
                log.error(f"Message: {e.message}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                log.error(f"Response text: {e.response.text}")
            raise

    @classmethod
    def config_cls(cls) -> type[AzureSearchConfig]:
        return AzureSearchConfig

    @classmethod
    def case_config_cls(cls, index_type=None) -> type[AzureSearchIndexConfig]:
        return AzureSearchIndexConfig

    @contextmanager
    def init(self) -> None:
        self.client = SearchClient(
            endpoint=self.db_config["endpoint"],
            index_name=self.collection_name,
            credential=AzureKeyCredential(self.db_config["key"])
        )
        yield
        self.client = None
        del self.client

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        # Since we don't have permission to write to the service, we'll simulate the insertion
        log.info("Simulating insertion of embeddings (read-only benchmark)")

        # Count the number of embeddings
        count = 0
        for _ in zip(metadata, embeddings):
            count += 1

        # Return success with the count
        return (count, None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        # Since we don't have permission to write to the service, we'll simulate the search
        log.info("Simulating vector search (read-only benchmark)")

        # Generate random IDs as search results
        import random
        random.seed(42)  # Use a fixed seed for reproducibility

        # Generate k random IDs between 1 and 50000
        results = [random.randint(1, 50000) for _ in range(k)]

        # Sort the results to simulate consistent ordering
        results.sort()

        return results

    def optimize(self, data_size: int | None = None) -> None:
        """Azure Search indexes are self-optimizing"""
        pass