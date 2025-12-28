from pydantic import BaseModel
from typing import Optional, Literal

class SyncRun(BaseModel):
    run_id: str

class SyncWindowRequest(BaseModel):
    start_date: str
    end_date: str

class DiffRequest(BaseModel):
    left_id: Optional[str] = None
    right_id: Optional[str] = None

class StatusResponse(BaseModel):
    run_id: str
    status: Literal['running','succeeded','failed']
    started_at_utc: str
    finished_at_utc: Optional[str] = None
    error_message: Optional[str] = None
