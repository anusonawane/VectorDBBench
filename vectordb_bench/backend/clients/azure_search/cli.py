"""Azure AI Search client for VectorDBBench"""

from typing import Annotated

import click
from pydantic import SecretStr

# Import DB enum when it's updated with AzureSearch
# from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

class AzureSearchTypedDict(CommonTypedDict):
    """Azure AI Search benchmark parameters"""
    endpoint: Annotated[
        str,
        click.option("--endpoint", type=str, help="Azure Search service endpoint", required=True)
    ]
    key: Annotated[
        str,
        click.option("--key", type=str, help="Azure Search admin key", required=True)
    ]
    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice([m.name for m in MetricType]),
            default=MetricType.COSINE.name,
            help="Metric type for vector similarity"
        )
    ]
    ef_construction: Annotated[
        int,
        click.option("--ef-construction", type=int, default=400, help="ef construction parameter")
    ]
    m: Annotated[
        int,
        click.option("--m", type=int, default=16, help="M parameter")
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=500, help="ef search parameter")
    ]


@cli.command("azure-search")
@click_parameter_decorators_from_typed_dict(AzureSearchTypedDict)
def AzureSearch(**parameters):
    """Azure AI Search vector database"""
    from vectordb_bench.backend.clients.azure_search.config import AzureSearchConfig, AzureSearchIndexConfig

    run(
        db="AzureSearch",
        db_config=AzureSearchConfig(
            endpoint=parameters["endpoint"],
            key=SecretStr(parameters["key"]),
        ),
        db_case_config=AzureSearchIndexConfig(
            metric_type=MetricType[parameters["metric_type"]],
            ef_construction=parameters["ef_construction"],
            m=parameters["m"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
