# app/schemas/chapter_material_schema.py

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator, model_validator


class TopicExtractRequest(BaseModel):
    material_ids: List[int]


class LectureConfigRequest(BaseModel):
    merged_id: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    duration: Optional[int] = Field(default=None, ge=5, le=180)
    model: Optional[str] = Field(default=None)


class LectureGenerationRequest(BaseModel):
    material_id: Optional[int] = Field(default=None)
    merged_lecture_id: Optional[str] = Field(default=None, alias="lecture_id")
    selected_topic_ids: Optional[List[str]] = Field(default=None, min_items=1)
    style: Optional[str] = Field(default="storytelling")
    language: Optional[str] = Field(default=None)
    duration: Optional[int] = Field(default=None, ge=5, le=180)
    model: Optional[str] = Field(default=None)

    @validator("selected_topic_ids")
    def ensure_topic_ids_present(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("At least one topic ID must be provided")
        cleaned = []
        for tid in value:
            tid_str = str(tid).strip()
            if not tid_str:
                raise ValueError("Topic IDs cannot be empty")
            cleaned.append(tid_str)
        return cleaned

    @model_validator(mode="after")
    def ensure_source_context(cls, values: "LectureGenerationRequest") -> "LectureGenerationRequest":
        if values.merged_lecture_id:
            if values.material_id is not None or values.selected_topic_ids:
                raise ValueError("Provide either lecture_id or material/topic selection, not both")
            return values
        if values.material_id is None:
            raise ValueError("material_id is required when lecture_id is not provided")
        if not values.selected_topic_ids:
            raise ValueError("At least one topic ID must be provided when lecture_id is not provided")
        return values

    class Config:
        allow_population_by_field_name = True

class LectureLookupRequest(BaseModel):
    std: str = Field(..., description="Class/standard identifier")
    subject: str = Field(..., description="Subject name")
    chapter_title: str = Field(..., description="Chapter number or title")

    @validator("std", "subject", "chapter_title")
    def strip_and_validate(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Fields cannot be empty")
        return cleaned


class TopicSelection(BaseModel):
    material_id: int
    topic_indices: Optional[List[int]] = Field(default=None, min_items=1)
    topic_ids: Optional[List[str]] = Field(default=None, alias="topic_id")
    suggestion_topic_ids: Optional[List[str]] = Field(
        default=None,
        alias="suggestion_topic_id",
    )

    @validator("topic_indices")
    def ensure_valid_indices(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if not value:
            raise ValueError("topic_indices cannot be empty")
        if any(index < 0 for index in value):
            raise ValueError("Topic indices must be zero-based positive integers")
        return value

    @model_validator(mode="after")
    def ensure_selection_present(cls, values: "TopicSelection") -> "TopicSelection":
        indices = values.topic_indices or []
        ids = values.topic_ids or []
        suggestion_ids = values.suggestion_topic_ids or []
        if not indices and not ids and not suggestion_ids:
            raise ValueError("Provide topic_indices or topic_ids for each selection")
        return values

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "material_id": 40,
                "topic_indices": [0, 2, 5],
            }
        }


class CreateMergedLectureRequest(BaseModel):
    title: Optional[str] = None
    materials: Optional[List[Dict[str, Any]]] = None
    topic_selections: Optional[List[TopicSelection]] = Field(default=None, alias="topicSelections")
    selected_topics: Optional[Dict[str, List[Dict[str, Any]]]] = Field(default=None, alias="selectedTopics")
    material_ids: Optional[List[int]] = None
    topic_filters: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    language: Optional[str] = Field(default=None)
    duration: Optional[int] = Field(default=None, ge=5, le=180)

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "topicSelections": [
                    {
                        "material_id": 40,
                        "topic_id": ["1", "2"]
                    },
                    {
                        "material_id": 41,
                        "topic_id": ["sg-101", "sg-205"]
                    }
                ]
            }
        }


class ManualTopicCreate(BaseModel):
    title: str
    summary: Optional[str] = None
    subtopics: Optional[List[Dict[str, str]]] = None


class SubtopicCreate(BaseModel):
    title: Optional[str] = None
    narration: Optional[str] = None


class AssistantSuggestRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    plan_label: Optional[Literal["20k", "50k", "100k"]] = Field(
        default=None,
        description="Optional override for the assistant suggestion plan tier",
    )


class AssistantAddTopicsRequest(BaseModel):
    selected_suggestions: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="(Optional) Full suggestion payloads to append directly",
    )
    suggestion_ids: Optional[List[str]] = Field(
        default=None,
        alias="suggestion_id",
        description="List of cached assistant suggestion IDs to add",
    )
    suggestion_topic_ids: Optional[List[str]] = Field(
        default=None,
        description="Alias for suggestion_ids when referring to suggested topic IDs",
    )

    @model_validator(mode="after")
    def ensure_selection_present(cls, values: "AssistantAddTopicsRequest") -> "AssistantAddTopicsRequest":
        suggestions = values.selected_suggestions or []
        suggestion_ids = (values.suggestion_ids or []) + (values.suggestion_topic_ids or [])
        values.suggestion_ids = values.suggestion_ids or []
        if values.suggestion_topic_ids:
            values.suggestion_ids.extend(str(sid) for sid in values.suggestion_topic_ids)
            values.suggestion_topic_ids = None
        if not suggestions and not values.suggestion_ids:
            raise ValueError("Provide either selected_suggestions or suggestion_ids")
        if values.suggestion_ids and any(not str(sid).strip() for sid in values.suggestion_ids):
            raise ValueError("Suggestion IDs cannot be empty")
        return values

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "suggestion_id": ["1", "3"],
            }
        }


class ChapterMaterialCreate(BaseModel):
    """Schema for creating a chapter material"""
    std: str
    subject: str
    sem: Optional[str] = ""
    board: Optional[str] = ""
    chapter_number: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "std": "10",
                "subject": "Mathematics",
                "sem": "1",
                "board": "CBSE",
                "chapter_number": "1"
            }
        }


class ChapterMaterialResponse(BaseModel):
    """Schema for chapter material response"""
    id: int
    admin_id: int
    std: str
    subject: str
    sem: Optional[str]
    board: Optional[str]
    chapter_number: str
    file_name: str
    file_path: str
    file_size: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "admin_id": 1,
                "std": "10",
                "subject": "Mathematics",
                "sem": "1",
                "board": "CBSE",
                "chapter_number": "1",
                "file_name": "chapter1.pdf",
                "file_path": "/uploads/chapter_materials/admin_1/chapter1.pdf",
                "file_size": 1024000,
                "created_at": "2025-01-15T10:30:00",
                "updated_at": "2025-01-15T10:30:00"
            }
        }


class ResponseBase(BaseModel):
    """Base response schema"""
    status: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": True,
                "message": "Operation successful",
                "data": {}
            }
        }


class MultipleChapterSelectionRequest(BaseModel):
    """Schema for selecting multiple chapters"""
    selected_ids: List[int] = Field(..., min_items=1, description="List of selected chapter IDs")
    
    @validator("selected_ids")
    def validate_chapter_ids(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("At least one chapter must be selected")
        return value