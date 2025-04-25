from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

class AzureSearchConfig(DBConfig):
    endpoint: str
    key: SecretStr
    index_name: str = "vectordbbench"

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "key": self.key.get_secret_value(),
            "index_name": self.index_name
        }

class AzureSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    dimensions: int = 1536
    ef_construction: int = 400
    m: int = 16
    ef_search: int = 500

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dotProduct"
        return "cosine"

    def index_param(self) -> dict:
        return {
            "algorithm": "hnsw",
            "parameters": {
                "m": self.m,
                "efConstruction": self.ef_construction,
                "metric": self.parse_metric()
            }
        }

    def search_param(self) -> dict:
        return {
            "ef": self.ef_search
        }