from pydantic import BaseModel, Field, ValidationError


class SampleCreateRequest(BaseModel):
    sample: str = Field(..., min_length=1, description="The sample text to be trained.")


class TransformRequest(BaseModel):
    target: str = Field(
        ..., min_length=1, description="The target text to be transformed."
    )


class TransformResponse(BaseModel):
    task_id: str


class NormalResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    error: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    receipt_id: str = None
    error: str = None
