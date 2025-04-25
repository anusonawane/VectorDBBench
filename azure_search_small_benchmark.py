#!/usr/bin/env python3
"""
Small benchmark for Azure AI Search to test client reuse optimization.
"""

import os
import time
import numpy as np
import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Azure AI Search credentials
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "")

# Vector dimensions
VECTOR_DIM = 1536

# Number of vectors to insert
NUM_VECTORS = 1000

class AzureSearchBenchmark:
    """Azure AI Search benchmark implementation."""

    def __init__(self, endpoint, key, index_name, vector_dim=1536, drop_old=True):
        """Initialize the benchmark."""
        self.endpoint = endpoint
        self.key = key
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.id_field = "id"
        self.vector_field = "vector"

        # Create admin client for index management
        self.admin_client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
            connection_timeout=3600,  # 60 minutes timeout
            read_timeout=3600
        )
        
        # Create search client for document operations with increased timeout
        # This client will be reused for all search operations
        self.search_client = None

        # Delete existing index if requested
        if drop_old:
            try:
                log.info(f"Deleting existing index '{index_name}' if it exists...")
                self.admin_client.delete_index(index_name)
                log.info(f"Index '{index_name}' deleted.")
            except Exception as e:
                log.info(f"Index deletion skipped: {e}")

    def create_index(self, ef_construction=400, max_connections=10, ef_search=500, metric_type="cosine"):
        """Create a new Azure AI Search index with vector search capabilities."""
        log.info(f"Creating index '{self.index_name}'...")

        # Define the vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": max_connections,
                        "efConstruction": ef_construction,
                        "efSearch": ef_search,
                        "metric": metric_type
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )

        # Define the fields
        fields = [
            SearchField(
                name=self.id_field,
                type=SearchFieldDataType.String,
                key=True
            ),
            SearchField(
                name=self.vector_field,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self.vector_dim,
                vector_search_profile_name="vector-profile"
            )
        ]

        # Create the index
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        result = self.admin_client.create_or_update_index(index)
        log.info(f"Index '{self.index_name}' created successfully.")
        
        # Initialize the search client after creating the index
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.key),
            connection_timeout=3600,  # 60 minutes timeout
            read_timeout=3600
        )
        log.info(f"Search client initialized for index '{self.index_name}'")
        
        return result

    def insert_vectors(self, vectors, batch_size=100):
        """Insert vectors into the Azure AI Search index."""
        log.info(f"Inserting {len(vectors)} vectors into index '{self.index_name}'...")

        # Use the existing search client
        if self.search_client is None:
            log.error("Search client not initialized. Call create_index first.")
            raise ValueError("Search client not initialized")

        # Prepare documents
        documents = []
        for i, vector in enumerate(vectors):
            doc = {
                self.id_field: str(i),
                self.vector_field: vector.tolist()
            }
            documents.append(doc)

        # Insert in batches
        total_inserted = 0
        start_time = time.time()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.search_client.upload_documents(batch)
            total_inserted += len(batch)
            log.info(f"Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({total_inserted}/{len(documents)})")

        end_time = time.time()
        duration = end_time - start_time
        qps = len(vectors) / duration

        log.info(f"Inserted {total_inserted} vectors in {duration:.2f} seconds ({qps:.2f} vectors/second)")
        return {
            "load_duration": duration,
            "load_qps": qps
        }

    def search_vector(self, query_vector, k=100):
        """Search for similar vectors in the Azure AI Search index."""
        log.info(f"Searching for similar vectors with k={k}...")
        # Use the existing search client
        if self.search_client is None:
            log.error("Search client not initialized. Call create_index first.")
            raise ValueError("Search client not initialized")

        try:
            # Perform vector search
            log.info("Executing vector search query...")
            results = self.search_client.search(
                search_text=None,
                vector_queries=[{
                    "kind": "vector",  # Required parameter specifying the type of vector query
                    "vector": query_vector.tolist(),
                    "k": k,
                    "fields": self.vector_field
                }],
                select=[self.id_field]
            )
            log.info("Vector search query executed successfully.")

            # Extract document IDs
            log.info("Extracting document IDs from search results...")
            doc_ids = [int(doc[self.id_field]) for doc in results]
            log.info(f"Found {len(doc_ids)} results.")
            return doc_ids
        except Exception as e:
            log.error(f"Error during vector search: {e}")
            # Return empty list on error
            return []

    def benchmark_search(self, num_queries=10, k=10):
        """Benchmark search performance."""
        log.info(f"Running search benchmark with {num_queries} queries, k={k}...")

        # Generate random query vectors
        query_vectors = [np.random.rand(self.vector_dim).astype(np.float32) for _ in range(num_queries)]

        # Measure search performance
        latencies = []

        for i, query_vector in enumerate(query_vectors):
            start_time = time.time()
            result = self.search_vector(query_vector, k)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            if (i+1) % 5 == 0:
                log.info(f"Completed {i+1}/{num_queries} queries")

        # Calculate metrics
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        qps = 1.0 / avg_latency

        log.info(f"Search benchmark completed: QPS={qps:.2f}, Latency={avg_latency*1000:.2f}ms, P99={p99_latency*1000:.2f}ms")

        return {
            "qps": qps,
            "latency": avg_latency,
            "p99_latency": p99_latency
        }

def generate_vectors(num_vectors, dim):
    """Generate random vectors for benchmarking."""
    log.info(f"Generating {num_vectors} random vectors with dimension {dim}...")
    return [np.random.rand(dim).astype(np.float32) for _ in range(num_vectors)]

def main():
    """Run a small Azure AI Search benchmark."""
    log.info("Running small Azure AI Search benchmark with client reuse optimization")

    # Check if Azure Search credentials are available
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        log.error("Azure Search credentials not found. Please set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY environment variables.")
        return

    # Parameters
    k = 10
    num_queries = 10

    # Index parameters
    index_params = {
        "efConstruction": 400,  # Azure AI Search requires this to be between 100 and 1000
        "maxConnection": 10,  # Azure AI Search limits this to 4-10
        "ef_search": 500  # Azure AI Search requires this to be between 100 and 1000
    }

    # Initialize benchmark
    benchmark = AzureSearchBenchmark(
        endpoint=AZURE_SEARCH_ENDPOINT,
        key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        vector_dim=VECTOR_DIM
    )

    # Create index
    log.info("Creating Azure AI Search index...")
    benchmark.create_index(
        ef_construction=index_params["efConstruction"],
        max_connections=index_params["maxConnection"],
        ef_search=index_params["ef_search"]
    )

    # Generate and load vectors
    log.info("Generating and loading vectors...")
    vectors = generate_vectors(NUM_VECTORS, VECTOR_DIM)
    load_metrics = benchmark.insert_vectors(vectors)
    log.info(f"Load duration: {load_metrics['load_duration']:.2f}s, Load QPS: {load_metrics['load_qps']:.2f}")

    # Run serial search benchmark
    log.info("Running serial search benchmark...")
    serial_metrics = benchmark.benchmark_search(num_queries, k)
    log.info(f"Serial search results: QPS={serial_metrics['qps']:.2f}, Latency={serial_metrics['latency']*1000:.2f}ms, P99={serial_metrics['p99_latency']*1000:.2f}ms")

    log.info("\nBenchmark complete!")

if __name__ == "__main__":
    main()
