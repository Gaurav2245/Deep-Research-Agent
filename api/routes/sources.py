"""Source management and scoring routes."""
from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, Research, Source
from database.source_scorer import SourceScorer, SourceFilter
from api.schemas import SourceScoringRequest, SourceScoringResponse, SourceInfo
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/research/{research_id}/sources", response_model=List[SourceInfo])
async def get_research_sources(
    research_id: UUID,
    filter_by_score: float = 0.0,
    db: Session = Depends(get_db)
):
    """Get all sources for a research session, optionally filtered by score."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    sources = research.sources
    
    # Filter by minimum score if specified
    if filter_by_score > 0:
        sources = [s for s in sources if s.source_score >= filter_by_score]
    
    return [
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
        for s in sources
    ]


@router.post("/score-source", response_model=SourceScoringResponse)
async def score_source(request: SourceScoringRequest):
    """
    Score a source based on reliability, relevance, and recency.
    
    This is a stateless operation - doesn't require stored research context.
    """
    try:
        url = str(request.url)
        
        scores = SourceScorer.calculate_source_score(
            url=url,
            raw_search_score=request.relevance_score or 0.5,
            content=request.content or "",
            query=request.query or request.title,
            title=request.title,
        )
        
        return SourceScoringResponse(
            url=url,
            overall_score=scores["overall_score"],
            authority=scores["authority"],
            freshness=scores["freshness"],
            relevance=scores["relevance"],
            content_quality=scores["content_quality"],
        )
    except Exception as e:
        logger.error(f"Error scoring source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/{research_id}/sources/best")
async def get_best_sources(
    research_id: UUID,
    top_n: int = 10,
    min_score: float = 0.5,
    db: Session = Depends(get_db)
):
    """
    Get the best/most reliable sources for a research session.
    
    Uses source scoring and filtering to select top sources.
    """
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    # Convert to dict format for filtering
    source_dicts = [
        {
            "id": s.id,
            "url": s.url,
            "title": s.title,
            "overall_score": s.source_score,
            "domain_authority": s.authority_score,
            "content": s.content,
        }
        for s in research.sources
    ]
    
    # Select best sources
    best = SourceFilter.select_best_sources(
        source_dicts,
        count=top_n,
        min_score=min_score
    )
    
    source_ids = [s["id"] for s in best]
    best_sources = db.query(Source).filter(Source.id.in_(source_ids)).all()
    
    return {
        "research_id": research_id,
        "total_sources": len(research.sources),
        "selected_sources": len(best_sources),
        "selection_criteria": {
            "top_n": top_n,
            "min_score": min_score,
        },
        "sources": [
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
            for s in best_sources
        ]
    }


@router.get("/research/{research_id}/sources/analysis")
async def get_source_analysis(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Get analysis of sources: diversity, quality, distribution."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    sources = research.sources
    
    # Domain analysis
    domains = {}
    for source in sources:
        domain = SourceScorer.extract_domain(source.url)
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(source)
    
    # Score distribution
    score_ranges = {
        "excellent": len([s for s in sources if s.source_score >= 0.8]),
        "good": len([s for s in sources if 0.6 <= s.source_score < 0.8]),
        "fair": len([s for s in sources if 0.4 <= s.source_score < 0.6]),
        "poor": len([s for s in sources if s.source_score < 0.4]),
    }
    
    # Authority distribution
    primary_sources = len([s for s in sources if s.is_primary_source])
    verified_sources = len([s for s in sources if s.is_verified])
    
    return {
        "research_id": research_id,
        "total_sources": len(sources),
        "domain_analysis": {
            "unique_domains": len(domains),
            "domains": {domain: len(srcs) for domain, srcs in domains.items()},
        },
        "score_distribution": score_ranges,
        "authority": {
            "primary_sources": primary_sources,
            "verified_sources": verified_sources,
        },
        "quality_metrics": {
            "avg_source_score": sum(s.source_score for s in sources) / len(sources) if sources else 0,
            "avg_authority": sum(s.authority_score for s in sources) / len(sources) if sources else 0,
            "avg_recency": sum(s.recency_score for s in sources) / len(sources) if sources else 0,
        }
    }


@router.put("/sources/{source_id}/verify")
async def mark_source_verified(
    source_id: UUID,
    verified: bool = True,
    db: Session = Depends(get_db)
):
    """Mark a source as manually verified."""
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    source.is_verified = verified
    db.commit()
    
    return {
        "source_id": source_id,
        "is_verified": source.is_verified,
        "url": source.url,
    }
