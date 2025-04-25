#!/usr/bin/env python3
"""
Run Azure Search benchmark directly.
"""

from vectordb_bench.backend.clients.azure_search.config import AzureSearchConfig, AzureSearchIndexConfig
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.cli.cli import run
from vectordb_bench.backend.cases import CaseType
from pydantic.v1 import SecretStr

def main():
    """Run Azure Search benchmark."""
    run(
        db="AzureSearch",
        db_config=AzureSearchConfig(
            endpoint="",
            key=SecretStr(""),
        ),
        db_case_config=AzureSearchIndexConfig(
            metric_type=MetricType.COSINE,
            ef_construction=400,
            m=16,
            ef_search=500,
        ),
        case_type="Performance1536D50K",
        # Skip loading data since we don't have write permissions
        load=False,
        drop_old=False,
        k=10,
        concurrency_duration=60,
        num_concurrency=[1, 2, 4, 8, 16],
        custom_dataset_config=None,
        custom_case_config=None,
        search_serial=True,
        search_concurrent=True,
        optimize=False,
        optimize_timeout=300,
        optimize_serial=True,
        optimize_concurrent=True,
        optimize_params={},
        warmup=True,
        warmup_count=100,
        dry_run=False,
    )

if __name__ == "__main__":
    main()
