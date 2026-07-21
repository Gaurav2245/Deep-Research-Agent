"""Conversation (chat history) management routes."""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, Conversation, Message, ConversationState, ConversationalKnowledge
from agents.orchestration import ConversationOrchestrator
from api.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationDetailResponse,
    MessageCreate,
    MessageResponse,
    ConversationQueryRequest,
    ConversationQueryResponse,
    ConversationMemoryStats,
    MemoryRetrievalInfo,
    ResolutionInfo,
    ExtractedFact,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new conversation (chat thread).
    
    Returns the conversation ID and metadata.
    """
    try:
        conversation = Conversation(
            title=request.title or "New Chat",
            user_id=request.user_id,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        # Initialize conversation state
        state = ConversationState(conversation_id=conversation.id)
        db.add(state)
        db.commit()
        
        logger.info(f"Created new conversation: {conversation.id}")
        return conversation
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(
    user_id: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get all conversations for a user (sidebar history).
    
    Returns conversations sorted by most recent first.
    """
    try:
        query = db.query(Conversation)
        
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        conversations = query.order_by(Conversation.updated_at.desc()).limit(limit).all()
        logger.info(f"Retrieved {len(conversations)} conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific conversation with all its messages.
    
    Returns full conversation history for context.
    """
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Manually build response with proper serialization
        messages = []
        if conversation.messages:
            for msg in conversation.messages:
                messages.append(MessageResponse(
                    id=msg.id,
                    conversation_id=msg.conversation_id,
                    role=msg.role,
                    content=msg.content,
                    research_id=msg.research_id,
                    context_data=msg.context_data,
                    created_at=msg.created_at
                ))
        
        response = ConversationDetailResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=conversation.message_count or 0,
            research_count=conversation.research_count or 0,
            messages=messages
        )
        
        logger.info(f"Retrieved conversation {conversation_id} with {len(messages)} messages")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/query", response_model=ConversationQueryResponse)
def query_conversation(
    conversation_id: UUID,
    request: ConversationQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a query in a conversation with layered memory support.
    
    Flow:
    1. Store user message
    2. Load conversation state
    3. Resolve follow-up references (pronouns, ellipsis)
    4. Retrieve from memory (ConversationalKnowledge)
    5. Decide: use memory only or perform external research
    6. Generate response
    7. Extract knowledge from response
    8. Store assistant message + facts
    
    Returns:
    - Assistant response
    - Memory coverage score
    - Resolved entities and references
    - Extracted facts
    - Whether external research was performed
    """
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Initialize orchestrator (coordinates all memory layers)
        orchestrator = ConversationOrchestrator(db)
        
        # STEP 1-4: begin_turn handles:
        # - Store user message
        # - Load/update conversation state
        # - Resolve follow-up references
        # - Retrieve from memory
        turn_ctx = orchestrator.begin_turn(
            conversation_id=conversation_id,
            user_query=request.query
        )
        
        # STEP 5: Get memory results for decision-making
        memory_results = orchestrator.get_memory_results()
        resolution_results = orchestrator.get_resolution_result()
        
        # Determine if we need to research
        should_research = (
            request.perform_research 
            if request.perform_research is not None 
            else orchestrator.should_research()
        )
        
        logger.info(
            f"Query received | conv={conversation_id} | "
            f"memory_coverage={memory_results['memory_coverage']:.1%} | "
            f"should_research={should_research}"
        )
        
        # STEP 6: Generate response
        # This is where you integrate with your existing research agent
        if should_research:
            # TODO: Integrate with your existing research graph/agent
            # For now, we'll use a placeholder that indicates research is needed
            assistant_response = generate_response_with_research(
                query=request.query,
                memory_context=memory_results.get("memory_context", ""),
                entities=resolution_results.get("active_entities", []),
            )
            research_performed = True
        else:
            # Answer directly from memory
            assistant_response = generate_response_from_memory(
                query=request.query,
                memory_context=memory_results.get("memory_context", ""),
                entities=memory_results.get("retrieved_entities", []),
            )
            research_performed = False
        
        # STEP 7-8: end_turn handles:
        # - Extract knowledge from response
        # - Store assistant message
        # - Update conversation metadata
        turn_ctx = orchestrator.end_turn(assistant_response)
        
        # Build response
        extracted_facts = [
            ExtractedFact(
                entity=f["entity"],
                attribute=f["attribute"],
                value=f["value"],
                confidence=f["confidence"],
            )
            for f in turn_ctx.extracted_facts
        ]
        
        memory_info = MemoryRetrievalInfo(
            should_use_memory=memory_results["should_use_memory"],
            memory_coverage=memory_results["memory_coverage"],
            retrieved_entities=memory_results["retrieved_entities"],
            num_facts=memory_results["memory_coverage"] * 10,  # Approximate
        )
        
        resolution_info = ResolutionInfo(
            is_follow_up=resolution_results["is_follow_up"],
            resolved_references=resolution_results["resolved_references"],
            primary_entity=resolution_results["primary_entity"],
            active_entities=resolution_results["active_entities"],
            resolution_confidence=resolution_results["resolution_confidence"],
        )
        
        return ConversationQueryResponse(
            assistant_message_id=turn_ctx.assistant_message_id,
            content=assistant_response,
            memory_info=memory_info,
            resolution_info=resolution_info,
            facts_extracted=extracted_facts,
            research_performed=research_performed,
            elapsed_ms=turn_ctx.elapsed_ms() or 0.0,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/memory-stats", response_model=ConversationMemoryStats)
def get_memory_stats(
    conversation_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get memory statistics and utilization for a conversation.
    
    Shows:
    - Number of turns
    - Entities tracked
    - Facts extracted
    - Memory efficiency (how often queries were answered from memory)
    """
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        orchestrator = ConversationOrchestrator(db)
        summary = orchestrator.get_conversation_summary(conversation_id)
        
        return ConversationMemoryStats(
            conversation_id=conversation_id,
            total_turns=summary.get("total_turns", 0),
            active_entities=summary.get("active_entities", []),
            unique_entities=summary.get("unique_entities", 0),
            facts_extracted=summary.get("facts_extracted", 0),
            top_topics=summary.get("top_topics", []),
            memory_efficiency=summary.get("memory_efficiency", 0.0),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory stats for {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS (integrate with your research agent)
# ============================================================================

def generate_response_from_memory(
    query: str,
    memory_context: str,
    entities: List[str],
) -> str:
    """
    Generate response using only memory (no external research).
    """
    if not memory_context:
        return f"I don't have information about '{query}' in our conversation yet."
    
    try:
        from llm.factory import create_llm
        from config import get_agent_config
        
        cfg = get_agent_config()
        llm = create_llm(cfg)
        
        prompt = f"""Based on the information from our conversation history below, answer this question: {query}

Conversation context (prior facts and answers):
{memory_context}

Entities in scope: {', '.join(entities) if entities else 'none'}

Answer the question directly using only the provided context. If the context doesn't fully answer the question, answer what you can. Do not speculate or add external information not found in the context."""
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error in generate_response_from_memory: {e}")
        return f"[MEMORY ERROR] I have the information but failed to format it: {memory_context[:200]}..."


def generate_response_with_research(
    query: str,
    memory_context: str,
    entities: List[str],
    conversation_id: UUID = None,
) -> str:
    """
    Generate response with external research.
    
    Called when memory coverage is insufficient (<0.7).
    """
    try:
        from main import run_research
        from agents.state import ResearchState
        
        logger.info(f"[API] Running research for: {query}")
        
        # We can pass the conversation_id and context to the research agent
        # so it has access to history and prior knowledge.
        
        # For now, let's keep it simple and just run the research graph
        result = run_research(query)
        
        if result.error:
            return f"I encountered an error while researching: {result.error}"
            
        return result.final_answer or "I couldn't find a clear answer to that."
        
    except Exception as e:
        logger.error(f"Error in generate_response_with_research: {e}", exc_info=True)
        return f"I'm sorry, I ran into an issue while researching that topic: {str(e)}"


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
def add_message(
    conversation_id: UUID,
    request: MessageCreate,
    db: Session = Depends(get_db)
):
    """
    Add a message to a conversation.
    
    Supports both user and assistant messages. Can attach research metadata.
    """
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Create message
        message = Message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            research_id=request.research_id,
            context_data=request.context_data,
        )
        
        db.add(message)
        
        # Update conversation message count and timestamp
        conversation.message_count = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count() + 1
        conversation.updated_at = __import__('datetime').datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        
        logger.info(f"Added {request.role} message to conversation {conversation_id}")
        return message
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message to conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def get_messages(
    conversation_id: UUID,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get messages from a conversation with pagination.
    
    Used for loading conversation history.
    """
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).offset(offset).limit(limit).all()
        
        logger.info(f"Retrieved {len(messages)} messages from conversation {conversation_id}")
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving messages from {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: UUID,
    title: str,
    db: Session = Depends(get_db)
):
    """
    Update conversation title (rename chat).
    """
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation.title = title
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Updated conversation {conversation_id} title to '{title}'")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a conversation and all its messages.
    """
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        db.delete(conversation)
        db.commit()
        
        logger.info(f"Deleted conversation {conversation_id}")
        return {"status": "deleted", "conversation_id": str(conversation_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
