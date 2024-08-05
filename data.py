from pydantic import BaseModel
from typing import List, Optional

class ProcessedResult(BaseModel):
    name: str
    distance: float

class SuccessResponse(BaseModel):
    status_code: int
    message: str
    result: Optional[List[ProcessedResult]] = None

class ErrorResponse(BaseModel):
    status_code: int
    message: str
    error: Optional[str] = None