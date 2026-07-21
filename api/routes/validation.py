"""Data validation and quality routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, Research
from database.data_validator import DataValidator
from api.schemas import DataQualityResponse, DataValidationResult
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/research/{research_id}/quality", response_model=DataQualityResponse)
async def get_data_quality(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive data quality assessment.
    
    Runs validation checks on research results.
    """
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    if not research.final_answer:
        raise HTTPException(
            status_code=400,
            detail="Research not yet complete - no answer to validate"
        )
    
    # Prepare source data
    sources = [
        {
            "url": s.url,
            "title": s.title,
            "content": s.content,
        }
        for s in research.sources
    ]
    
    # Run all validations
    validation_results = DataValidator.validate_all(
        final_answer=research.final_answer,
        sources=sources,
        query=research.query,
    )
    
    # Convert to response format
    validation_details = [
        DataValidationResult(
            validation_type=r["validation_type"],
            passed=r.get("passed", False),
            score=r.get("score", 0.0),
            issues=r.get("issues", []) + r.get("conflicts", []) + r.get("unsupported_claims", []),
        )
        for r in validation_results["results"]
    ]
    
    # Generate recommendations
    recommendations = []
    if not validation_results["all_passed"]:
        for result in validation_results["results"]:
            if not result.get("passed"):
                validation_type = result.get("validation_type", "unknown")
                if validation_type == "completeness":
                    recommendations.append("Consider conducting additional research iterations")
                elif validation_type == "consistency":
                    recommendations.append("Review conflicting information in sources")
                elif validation_type == "factual_claims":
                    recommendations.append("Verify unsupported claims against primary sources")
                elif validation_type == "hallucination":
                    recommendations.append("Review quoted content for accuracy")
    else:
        recommendations.append("Data quality checks passed successfully")
    
    # Update research record
    research.data_quality_score = validation_results["overall_quality_score"]
    research.hallucination_flagged = any(
        r.get("hallucination_flagged", False) for r in validation_results["results"]
    )
    db.commit()
    
    return DataQualityResponse(
        research_id=research_id,
        overall_quality_score=validation_results["overall_quality_score"],
        all_validations_passed=validation_results["all_passed"],
        validation_results=validation_details,
        issues_count=validation_results["issues_found"],
        recommendations=recommendations,
    )


@router.post("/validate/completeness")
async def validate_completeness(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Validate that research answer is complete."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    if not research.final_answer:
        raise HTTPException(status_code=400, detail="No answer to validate")
    
    sources = [
        {"url": s.url, "title": s.title, "content": s.content}
        for s in research.sources
    ]
    
    result = DataValidator.validate_completeness(
        final_answer=research.final_answer,
        sources=sources,
        query=research.query,
    )
    
    return result


@router.post("/validate/consistency")
async def validate_consistency(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Validate consistency across sources."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    if not research.final_answer:
        raise HTTPException(status_code=400, detail="No answer to validate")
    
    sources = [
        {"url": s.url, "title": s.title, "content": s.content}
        for s in research.sources
    ]
    
    result = DataValidator.validate_consistency(
        sources=sources,
        final_answer=research.final_answer,
    )
    
    return result


@router.post("/validate/hallucination")
async def detect_hallucination(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Detect potential hallucinations in answer."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    if not research.final_answer:
        raise HTTPException(status_code=400, detail="No answer to validate")
    
    sources = [
        {"url": s.url, "title": s.title, "content": s.content}
        for s in research.sources
    ]
    
    result = DataValidator.detect_hallucination_markers(
        final_answer=research.final_answer,
        sources=sources,
    )
    
    return result


@router.post("/validate/factual-claims")
async def validate_factual_claims(
    research_id: UUID,
    db: Session = Depends(get_db)
):
    """Validate that factual claims are supported by sources."""
    research = db.query(Research).filter(Research.id == research_id).first()
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    if not research.final_answer:
        raise HTTPException(status_code=400, detail="No answer to validate")
    
    sources = [
        {"url": s.url, "title": s.title, "content": s.content}
        for s in research.sources
    ]
    
    result = DataValidator.validate_factual_claims(
        final_answer=research.final_answer,
        sources=sources,
    )
    
    return result
