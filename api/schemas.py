"""Pydantic schemas for API requests/responses."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class SourceInfo(BaseModel):
    """Source information schema."""
    id: Optional[UUID] = None
    title: str
    url: HttpUrl
    content: Optional[str] = None
    source_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    authority_score: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    is_primary_source: bool = False
    content_quality: float = Field(ge=0, le=1)
    is_verified: bool = False
    discovered_at: Optional[datetime] = None


class ResearchRequest(BaseModel):
    """Request to start new research."""
    query: str = Field(min_length=5, max_length=500)
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")


class ResearchResponse(BaseModel):
    """Response with research results."""
    id: UUID
    query: str
    final_answer: Optional[str] = None
    confidence_score: float = Field(ge=0, le=1)
    data_quality_score: float = Field(ge=0, le=1)
    research_complete: bool
    total_iterations: int
    sources_count: int
    created_at: datetime
    updated_at: datetime


class ResearchDetailResponse(BaseModel):
    """Detailed research response with all data."""
    id: UUID
    query: str
    final_answer: Optional[str] = None
    confidence_score: float
    data_quality_score: float
    research_complete: bool
    total_iterations: int
    follow_up_questions: List[str]
    sources: List[SourceInfo]
    validation_issues: List[Dict[str, Any]]
    hallucination_flagged: bool
    created_at: datetime
    updated_at: datetime


class FollowUpQuestionsRequest(BaseModel):
    """Request to generate follow-up questions."""
    research_id: UUID


class FollowUpQuestionsResponse(BaseModel):
    """Response with follow-up questions."""
    research_id: UUID
    original_query: str
    follow_up_questions: List[str]
    suggested_next_search: bool


class ConfidenceScoreResponse(BaseModel):
    """Confidence scoring response."""
    research_id: UUID
    overall_confidence: float = Field(ge=0, le=1)
    source_diversity: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    data_consistency: float = Field(ge=0, le=1)
    answer_completeness: float = Field(ge=0, le=1)
    no_hallucination: float = Field(ge=0, le=1)
    should_continue: bool
    reason: str


class DataValidationResult(BaseModel):
    """Data validation check result."""
    validation_type: str
    passed: bool
    score: float = Field(ge=0, le=1)
    issues: List[str] = []


class DataQualityResponse(BaseModel):
    """Data quality assessment response."""
    research_id: UUID
    overall_quality_score: float = Field(ge=0, le=1)
    all_validations_passed: bool
    validation_results: List[DataValidationResult]
    issues_count: int
    recommendations: List[str]


class SourceScoringRequest(BaseModel):
    """Request for source scoring."""
    url: HttpUrl
    title: str
    content: Optional[str] = None
    relevance_score: Optional[float] = Field(default=0.5, ge=0, le=1)
    query: Optional[str] = None


class SourceScoringResponse(BaseModel):
    """Source scoring response."""
    url: HttpUrl
    overall_score: float = Field(ge=0, le=1)
    authority: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    content_quality: float = Field(ge=0, le=1)


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: int


class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=10000)
    research_id: Optional[UUID] = None
    context_data: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    research_id: Optional[UUID] = None
    context_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    title: Optional[str] = Field(default="New Chat", max_length=500)
    user_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """Schema for conversation response."""
    id: UUID
    user_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    research_count: int

    class Config:
        from_attributes = True


class ConversationDetailResponse(BaseModel):
    """Detailed conversation with all messages."""
    id: UUID
    user_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    research_count: int
    messages: List[MessageResponse]

    class Config:
        from_attributes = True


class ConversationQueryRequest(BaseModel):
    """Request to query a conversation with memory-aware retrieval."""
    query: str = Field(min_length=1, max_length=10000)
    conversation_id: UUID
    perform_research: Optional[bool] = None  # If None, auto-decide based on memory coverage


class MemoryRetrievalInfo(BaseModel):
    """Information about memory retrieval."""
    should_use_memory: bool
    memory_coverage: float = Field(ge=0, le=1)
    retrieved_entities: List[str] = []
    num_facts: int = 0


class ResolutionInfo(BaseModel):
    """Information about follow-up reference resolution."""
    is_follow_up: bool
    resolved_references: Dict[str, str] = {}
    primary_entity: Optional[str] = None
    active_entities: List[str] = []
    resolution_confidence: float = Field(ge=0, le=1)


class ExtractedFact(BaseModel):
    """A single extracted fact from conversation."""
    entity: str
    attribute: str
    value: str
    confidence: float = Field(ge=0, le=1)


class ConversationQueryResponse(BaseModel):
    """Response to a conversation query with memory and resolution info."""
    assistant_message_id: UUID
    content: str
    
    # Memory and resolution info
    memory_info: MemoryRetrievalInfo
    resolution_info: ResolutionInfo
    
    # Response metadata
    facts_extracted: List[ExtractedFact] = []
    research_performed: bool = False
    elapsed_ms: float
    
    class Config:
        from_attributes = True


class ConversationMemoryStats(BaseModel):
    """Memory statistics for a conversation."""
    conversation_id: UUID
    total_turns: int
    active_entities: List[str] = []
    unique_entities: int = 0
    facts_extracted: int = 0
    top_topics: List[str] = []
    memory_efficiency: float = Field(ge=0, le=1)
    
    class Config:
        from_attributes = True
