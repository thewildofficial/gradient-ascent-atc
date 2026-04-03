"""Task implementations for arrival, departure, and integrated benchmarks."""

from pydantic import BaseModel


class TaskRegistry(BaseModel):
    """Maps task IDs to their implementations."""

    pass
