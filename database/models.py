"""SQLAlchemy models for Deep Research Agent."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Text, Float, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, UUID
import uuid

Base = declarative_base()


class Research(Base):
    """Main research session record."""
    __tablename__ = "research"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(String(500), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Research metadata
    final_answer = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)  # 0.0-1.0: how confident are we?
    research_complete = Column(Boolean, default=False)
    
    # Iteration and follow-up tracking
    total_iterations = Column(Integer, default=0)
    follow_up_questions = Column(JSON, default=list)  # List of generated follow-ups
    
    # Data quality metadata
    data_quality_score = Column(Float, default=0.0)  # 0.0-1.0
    validation_issues = Column(JSON, default=list)  # List of issues found
    hallucination_flagged = Column(Boolean, default=False)
    chat_history = Column(JSON, default=list)  # List of {role, content}
    
    # v2.3 Conversational Cognition (NEW)
    understood_intent = Column(Text, nullable=True)
    query_reasoning = Column(Text, nullable=True)
    active_topic = Column(String(255), nullable=True)
    is_follow_up = Column(Boolean, default=False)
    entities_resolved = Column(JSON, default=dict)
    conversational_knowledge = Column(JSON, default=dict)  # entity → attributes (session reasoning)

    # Vector embedding
    embedding = Column(ARRAY(Float, as_tuple=False), nullable=True)
    embedding_model = Column(String(50), default="text-embedding-3-small")
    
    # Relationships
    sources = relationship("Source", back_populates="research", cascade="all, delete-orphan")
    validations = relationship("DataValidation", back_populates="research", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Research(id={self.id}, query='{self.query[:50]}...', confidence={self.confidence_score})>"


class Source(Base):
    """Web source (URL) discovered during research."""
    __tablename__ = "source"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_id = Column(UUID(as_uuid=True), ForeignKey("research.id"), nullable=False, index=True)
    
    # Source information
    title = Column(String(500), nullable=False)
    url = Column(String(2000), nullable=False, index=True)
    content = Column(Text, nullable=True)
    
    # Scoring and filtering
    source_score = Column(Float, default=0.0)  # 0.0-1.0: reliability/relevance
    relevance_score = Column(Float, default=0.0)  # Original Tavily score
    authority_score = Column(Float, default=0.0)  # Domain authority (0-1)
    recency_score = Column(Float, default=0.0)  # How recent is content (0-1)
    is_primary_source = Column(Boolean, default=False)
    
    # Data quality
    content_quality = Column(Float, default=0.0)  # 0-1: grammar, clarity, completeness
    contains_claims = Column(Boolean, default=False)  # Has factual claims
    is_verified = Column(Boolean, default=False)  # Manually verified
    
    # Scraping metadata
    scraped_at = Column(DateTime, nullable=True)
    scraped_successfully = Column(Boolean, default=False)
    scrape_method = Column(String(50), default="tavily")  # tavily, playwright, nse, etc.
    
    # Vector embedding
    content_embedding = Column(ARRAY(Float, as_tuple=False), nullable=True)
    
    # Timestamps
    discovered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    research = relationship("Research", back_populates="sources")
    scores = relationship("SourceScore", back_populates="source", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Source(url='{self.url[:50]}...', score={self.source_score})>"


class SourceScore(Base):
    """Detailed scoring breakdown for a source."""
    __tablename__ = "source_score"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("source.id"), nullable=False, index=True)
    
    # Scoring breakdown
    domain_authority = Column(Float, default=0.0)  # Is it a known reliable domain?
    content_freshness = Column(Float, default=0.0)  # How recent is the data?
    topical_relevance = Column(Float, default=0.0)  # How relevant to query?
    factual_consistency = Column(Float, default=0.0)  # Consistent with other sources?
    citation_quality = Column(Float, default=0.0)  # Sources cited, links provided?
    
    # Scoring metadata
    scorer_version = Column(String(50), default="v1")
    scoring_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source = relationship("Source", back_populates="scores")
    
    def __repr__(self) -> str:
        return f"<SourceScore(source_id={self.source_id}, total={self.calculate_average()})>"
    
    def calculate_average(self) -> float:
        """Calculate weighted average of all scores."""
        scores = [
            self.domain_authority,
            self.content_freshness,
            self.topical_relevance,
            self.factual_consistency,
            self.citation_quality,
        ]
        return sum(scores) / len(scores) if scores else 0.0


class DataValidation(Base):
    """Track data validation and quality assurance checks."""
    __tablename__ = "data_validation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_id = Column(UUID(as_uuid=True), ForeignKey("research.id"), nullable=False, index=True)
    
    # Validation metadata
    validation_type = Column(String(50), nullable=False)  # completeness, consistency, hallucination, etc.
    passed = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)  # How confident is this check?
    
    # Issue details
    issue_description = Column(Text, nullable=True)
    affected_claims = Column(JSON, default=list)  # List of flagged claims
    
    # Severity and action
    severity = Column(String(20), default="warning")  # critical, warning, info
    requires_review = Column(Boolean, default=False)
    resolution = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    research = relationship("Research", back_populates="validations")
    
    def __repr__(self) -> str:
        return f"<DataValidation(type={self.validation_type}, passed={self.passed}, severity={self.severity})>"


class Conversation(Base):
    """Chat conversation thread (ChatGPT-like history)."""
    __tablename__ = "conversation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=True, index=True)  # For multi-user support
    title = Column(String(500), nullable=False, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    research_count = Column(Integer, default=0)  # How many research sessions in this chat?
    message_count = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', messages={self.message_count})>"


class Message(Base):
    """Individual message in a conversation."""
    __tablename__ = "message"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id"), nullable=False, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    # Research metadata (if this message is part of research)
    research_id = Column(UUID(as_uuid=True), ForeignKey("research.id"), nullable=True)
    
    # Message context data (sources, confidence, iteration data, etc.)
    context_data = Column(JSON, nullable=True)
    
    # Vector embedding for semantic search (stored as ARRAY for PostgreSQL)
    embedding = Column(ARRAY(Float, as_tuple=False), nullable=True)
    embedding_model = Column(String(50), default="text-embedding-3-small")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, preview='{self.content[:30]}...')>"


class ConversationState(Base):
    """
    Short-term working memory for a conversation.
    
    Stores the active cognitive state: what are we talking about right now?
    Updated after each user turn, used for follow-up resolution and context.
    """
    __tablename__ = "conversation_state"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id"), nullable=False, unique=True, index=True)
    
    # Active cognition
    active_topic = Column(String(500), nullable=True)  # Primary subject being discussed
    current_focus = Column(String(500), nullable=True)  # What aspect are we examining?
    last_intent = Column(Text, nullable=True)  # What was the user trying to do?
    
    # Active entities in scope for this conversation
    active_entities = Column(JSON, default=list)  # ["Heinrich Klaasen", "KL Rahul", ...]
    recent_entity_mentions = Column(JSON, default=dict)  # {entity: count}
    
    # Working memory for context
    working_memory = Column(JSON, default=dict)  # Arbitrary key-value store
    
    # Scope context: what domain/season/tournament?
    scope_context = Column(String(500), nullable=True)  # e.g., "IPL 2026", "Q1 2024"
    
    # Disambiguation hints
    pending_references = Column(JSON, default=dict)  # {pronoun: resolved_entity}
    
    # Metadata
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    turn_count = Column(Integer, default=0)  # How many exchanges in this conversation?
    
    # Relationships
    conversation = relationship("Conversation", backref="state", uselist=False)
    
    def __repr__(self) -> str:
        return f"<ConversationState(conv_id={self.conversation_id}, topic='{self.active_topic}')>"


class ConversationalKnowledge(Base):
    """
    Relational knowledge extracted from conversation.
    
    MOST IMPORTANT table: powers follow-up answers and context.
    
    Instead of storing entire messages, we extract and store structured facts:
    - Entity: "Heinrich Klaasen"
    - Attribute: "strike_rate"
    - Value: 153.93
    - Confidence: 0.95
    - Source message
    
    This enables:
    1. Follow-up answer directly from memory (no re-search)
    2. Relationship queries ("all mentioned batsmen")
    3. Contradiction detection
    4. Conversation summarization
    """
    __tablename__ = "conversational_knowledge"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id"), nullable=False, index=True)
    
    # Extracted fact: entity → attribute → value
    entity = Column(String(500), nullable=False, index=True)  # e.g., "Heinrich Klaasen"
    attribute = Column(String(500), nullable=False, index=True)  # e.g., "strike_rate"
    value = Column(Text, nullable=False)  # e.g., "153.93" or JSON for complex values
    
    # Typing: what kind of value?
    value_type = Column(String(50), default="string")  # string, number, date, list, dict, etc.
    
    # Quality
    confidence = Column(Float, default=0.9)  # 0.0-1.0: how confident is this fact?
    extraction_method = Column(String(50), default="llm")  # llm, regex, manual, etc.
    
    # Provenance
    source_message_id = Column(UUID(as_uuid=True), ForeignKey("message.id"), nullable=True)
    source_text_snippet = Column(Text, nullable=True)  # The exact quote from source
    
    # Temporal tracking
    first_mentioned_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Deprecation: mark facts as stale/contradicted
    is_active = Column(Boolean, default=True)  # Might be superseded by newer facts
    superseded_by = Column(UUID(as_uuid=True), nullable=True)  # If superseded, by which id?
    
    # Relationships
    conversation = relationship("Conversation", backref="knowledge_base")
    source_message = relationship("Message", foreign_keys=[source_message_id])
    
    def __repr__(self) -> str:
        return f"<ConversationalKnowledge({self.entity}.{self.attribute}={self.value}, conf={self.confidence:.2f})>"
