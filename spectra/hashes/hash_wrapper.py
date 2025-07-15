from pydantic import BaseModel, Field
from typing import Callable

from .PDQ import PDQHasher


class Hash_Wrapper(BaseModel):
    """Parameterized wrapper for hash functions"""
    name: str = Field(..., description="Name of the hash function")
    func: Callable = Field(..., description="Hash function")
    resize_height: int = Field(default=-1, description="Resize height")
    resize_width: int = Field(default=-1, description="Resize width")
    available_devices: set[str] = Field(default={"cpu"}, description="Available devices")
