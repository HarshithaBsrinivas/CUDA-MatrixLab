"""Schema for benchmark result documents (the JSON produced by the Colab notebook
or by local runs). The same shape is used everywhere in the project."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

KernelName = Literal[
    "cpu_matmul",
    "cuda_naive_matmul",
    "cuda_shared_matmul",
    "cuda_vector_add",
]


class Metric(BaseModel):
    type: Literal["gflops", "gbps"]
    value: Optional[float] = None


class RunRecord(BaseModel):
    kernel: KernelName
    n: int = Field(gt=0)
    reps: int = Field(gt=0)
    device: str
    compute_capability: Optional[str] = None
    time_scope: str
    times_ms: list[float] = Field(min_length=1)
    median_ms: float
    min_ms: float
    metric: Metric
    verified: bool
    max_abs_error: Optional[float] = None


class ResultsDocument(BaseModel):
    # The JSON key is "schema"; the attribute is renamed to avoid clashing
    # with pydantic's BaseModel.schema method.
    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["cuda-matrixlab-results/v1"] = Field(alias="schema")
    generated_at: str
    source: str
    environment: dict = Field(default_factory=dict)
    runs: list[RunRecord] = Field(min_length=1)

    def to_json_dict(self) -> dict:
        return self.model_dump(by_alias=True, mode="json")
