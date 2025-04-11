from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AskResponse(BaseModel):
    answer: str
