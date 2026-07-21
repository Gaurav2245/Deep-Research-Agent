"""Research session management routes."""
from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db, Research
from database.confidence_scorer import ConfidenceScorer
from database.data_validator import DataValidator
from api.schemas import (
    ResearchRequest,
    ResearchResponse,
    ResearchDetailResponse,
    FollowUpQuestionsResponse,
    ConfidenceScoreResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a new research session.
    
    Returns the research session ID immediately.
    Processing happens in the background.
    """
    try:
        # Create research record
        research = Research(
            query=request.query,
            research_complete=False,
            confidence_score=0.0,
            data_quality_score=0.0,
        )
        db.add(research)
        db.commit()
        db.refresh(research)
        
        # Add background task to run research
        background_tasks.add_task(
            _run_research_async,
            research_id=research.id,
            query=request.query,
            depth=request.depth
        )
        
        logger.info(f"Started research session: {research.id} - Query: {request.query}")
        
        return ResearchResponse(
            id=research.id,
            query=research.query,
            final_answer=None,
            confidence_score=0.0,
            data_quality_score=0.0,
            research_complete=False,
            total_iterations=0,
            sources_count=0,
            created_at=research.created_at,
            updated_at=research.updated_at,
        )
    except Exception as e:
        logger.error(f"Error starting research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/{research_id}", response_model=ResearchResponse)
async def get_research_status(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Get status of a research session."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    return ResearchResponse(
        id=research.id,
        query=research.query,
        final_answer=research.final_answer,
        confidence_score=research.confidence_score,
        data_quality_score=research.data_quality_score,
        research_complete=research.research_complete,
        total_iterations=research.total_iterations,
        sources_count=len(research.sources),
        created_at=research.created_at,
        updated_at=research.updated_at,
    )


@router.get("/research/{research_id}/detail", response_model=ResearchDetailResponse)
async def get_research_detail(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Get detailed research results including sources and validation."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    from api.schemas import SourceInfo
    
    sources = [
        SourceInfo(
            id=s.id,
            title=s.title,
            url=str(s.url),
            content=s.content[:500] if s.content else None,
            source_score=s.source_score,
            relevance_score=s.relevance_score,
            authority_score=s.authority_score,
            recency_score=s.recency_score,
            is_primary_source=s.is_primary_source,
            content_quality=s.content_quality,
            is_verified=s.is_verified,
            discovered_at=s.discovered_at,
        )
        for s in research.sources
    ]
    
    validation_issues = [
        {
            "type": v.validation_type,
            "passed": v.passed,
            "severity": v.severity,
            "description": v.issue_description,
        }
        for v in research.validations
    ]
    
    return ResearchDetailResponse(
        id=research.id,
        query=research.query,
        final_answer=research.final_answer,
        confidence_score=research.confidence_score,
        data_quality_score=research.data_quality_score,
        research_complete=research.research_complete,
        total_iterations=research.total_iterations,
        follow_up_questions=research.follow_up_questions or [],
        sources=sources,
        validation_issues=validation_issues,
        hallucination_flagged=research.hallucination_flagged,
        created_at=research.created_at,
        updated_at=research.updated_at,
    )


@router.get("/research/{research_id}/confidence", response_model=ConfidenceScoreResponse)
async def get_confidence_score(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Get confidence score breakdown."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    # Prepare data for confidence calculation
    sources = [
        {
            "url": s.url,
            "overall_score": s.source_score,
            "content": s.content,
        }
        for s in research.sources
    ]
    
    embeddings = [s.embedding for s in research.sources if s.embedding]
    
    # Calculate confidence
    confidence_data = ConfidenceScorer.calculate_overall_confidence(
        sources=sources,
        final_answer=research.final_answer or "",
        query=research.query,
        embeddings=embeddings,
    )
    
    should_continue, reason = ConfidenceScorer.should_continue_research(
        confidence_data["overall_confidence"],
        iterations=research.total_iterations,
    )
    
    return ConfidenceScoreResponse(
        research_id=research_id,
        overall_confidence=confidence_data["overall_confidence"],
        source_diversity=confidence_data["source_diversity"],
        source_quality=confidence_data["source_quality"],
        data_consistency=confidence_data["data_consistency"],
        answer_completeness=confidence_data["answer_completeness"],
        no_hallucination=confidence_data["no_hallucination"],
        should_continue=should_continue,
        reason=reason,
    )


@router.get("/research/{research_id}/followups", response_model=FollowUpQuestionsResponse)
async def get_follow_up_questions(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Get generated follow-up questions."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    return FollowUpQuestionsResponse(
        research_id=research_id,
        original_query=research.query,
        follow_up_questions=research.follow_up_questions or [],
        suggested_next_search=not research.research_complete,
    )


@router.get("/research")
async def list_research_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List all research sessions."""
    query = db.query(Research).order_by(Research.created_at.desc())
    total = query.count()
    
    results = query.offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "sessions": [
            ResearchResponse(
                id=r.id,
                query=r.query,
                final_answer=r.final_answer,
                confidence_score=r.confidence_score,
                data_quality_score=r.data_quality_score,
                research_complete=r.research_complete,
                total_iterations=r.total_iterations,
                sources_count=len(r.sources),
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in results
        ]
    }


async def _run_research_async(research_id: UUID, query: str, depth: str):
    """
    Background task to run research.
    
    This would integrate with your existing LangGraph agent.
    """
    try:
        from main import run_research
        from database import SessionLocal
        
        # Run research using existing agent
        result = run_research(query)
        
        # Update database with results
        db = SessionLocal()
        research = db.query(Research).filter(Research.id == research_id).first()
        
        if research:
            research.final_answer = result.final_answer
            research.follow_up_questions = result.follow_up_questions
            research.total_iterations = result.iteration
            research.research_complete = True
            
            # TODO: Run validation and scoring
            # TODO: Store sources with scoring
            # TODO: Calculate confidence
            
            db.commit()
            logger.info(f"Research completed: {research_id}")
        
        db.close()
    except Exception as e:
        logger.error(f"Error in background research: {e}", exc_info=True)
