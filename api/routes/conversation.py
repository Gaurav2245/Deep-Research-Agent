"""Conversation (chat history) management routes."""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, Conversation, Message, ConversationState
from api.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationDetailResponse,
    MessageCreate,
    MessageResponse,
    ConversationQueryRequest,
    ConversationQueryResponse,
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
    Process a query in a conversation.

    Stores the user message, runs the research agent, stores the assistant
    reply, and returns it.
    """
    start = datetime.utcnow()
    try:
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        db.add(Message(conversation_id=conversation_id, role="user", content=request.query))

        assistant_content = _run_research_for_query(request.query)

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
        )
        db.add(assistant_message)

        conversation.message_count = (conversation.message_count or 0) + 2
        conversation.research_count = (conversation.research_count or 0) + 1
        conversation.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(assistant_message)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        return ConversationQueryResponse(
            assistant_message_id=assistant_message.id,
            content=assistant_content,
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _run_research_for_query(query: str) -> str:
    """Run the research agent for a conversational query and return its answer text."""
    try:
        from main import run_research

        logger.info(f"[API] Running research for: {query}")
        result = run_research(query)

        if result.error:
            return f"I encountered an error while researching: {result.error}"

        return result.final_answer or "I couldn't find a clear answer to that."

    except Exception as e:
        logger.error(f"Error running research for query: {e}", exc_info=True)
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
        conversation.updated_at = datetime.utcnow()
        
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
