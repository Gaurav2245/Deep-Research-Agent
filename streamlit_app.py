"""
streamlit_app.py
Advanced Streamlit UI for Deep Research Agent with real-time process visualization.
Shows entire research journey with complete source tracking.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import time
import os
import sys
import threading

# Load environment variables early
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

try:
    from agents import ResearchState, build_research_graph
    from agents.state import ResearchState as StateClass
    from config import get_agent_config
    from logging_handler import setup_logging_capture, get_streamlit_handler
    from utils.logger import get_logger
    from utils.pdf_generator import generate_pdf_from_state
    from database import SessionLocal, Research, Source, Conversation, Message, init_db
    from utils.conversation_client import ConversationClient
except Exception as e:
    st.error(f"❌ Failed to import modules: {str(e)}")
    st.error("This usually means a required environment variable is missing.")
    st.info("Please check:")
    st.code("1. AZURE_OPENAI_API_KEY is set in .env\n2. TAVILY_API_KEY is set in .env\n3. All dependencies are installed", language="text")
    st.stop()

logger = get_logger(__name__)
conversation_client = ConversationClient()

# Ensure DB schema matches models (adds missing columns; create_all does not ALTER).
if "_deep_research_db_init" not in st.session_state:
    st.session_state._deep_research_db_init = True
    try:
        init_db()
    except Exception as db_exc:
        logger.warning("Database init/schema patch failed: %s", db_exc)

# Database utilities

def create_new_conversation(title: str = "New Chat", preserve_research_state: bool = False):
    """Create a new conversation and set it as active."""
    logger.info(f"Creating new conversation with title: {title}")
    convo = conversation_client.create_conversation(title=title, user_id="default_user")
    if convo:
        st.session_state.current_conversation_id = str(convo["id"])
        st.session_state.current_conversation_title = convo.get("title", title)
        st.session_state.conversation_messages = []  # Empty conversation, no messages yet
        
        # Only clear history if we're not preserving research state (which needs history)
        if not preserve_research_state:
            st.session_state.history = []  # Clear research agent history
            st.session_state.research_state = None  # Clear old research results
        
        st.session_state.current_research_id = None  # Clear research ID
        logger.info(f"Created new conversation: {convo['id']}")
        return convo
    else:
        logger.error("Failed to create conversation - API returned None")
        st.error("Failed to create new chat. Please check if API is running.")
        return None


def load_conversation(conversation_id: str):
    """Load a specific conversation."""
    convo = conversation_client.get_conversation(conversation_id)
    if convo:
        st.session_state.current_conversation_id = conversation_id
        st.session_state.current_conversation_title = convo.get("title", "Chat")
        # Load messages from the conversation
        messages = convo.get("messages", [])
        st.session_state.conversation_messages = messages
        
        # Populate history for the research agent from conversation messages
        st.session_state.history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages 
            if msg["role"] in ("user", "assistant")
        ]
        
        st.session_state.research_state = None  # Clear old research results
        st.session_state.current_research_id = None  # Clear research ID
        
        # Auto-load research from the latest assistant message with research_id
        # Only if we're not in an active research session (prevent mid-conversation interference)
        if not st.session_state.is_researching:
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("research_id"):
                    logger.info(f"Auto-loading research {msg['research_id']} from latest message")
                    if load_research_to_state(msg["research_id"]):
                        st.session_state.current_research_id = msg["research_id"]
                        break
        
        logger.info(f"Loaded conversation: {conversation_id} with {len(messages)} messages")
        return convo
    return None


def get_all_conversations():
    """Fetch all conversations for the sidebar."""
    logger.info("Fetching all conversations...")
    try:
        result = conversation_client.list_conversations(user_id="default_user", limit=50)
        logger.info(f"Got {len(result)} conversations")
        return result
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        st.error(f"Could not fetch chat history: {e}")
        return []


def save_message_to_conversation(role: str, content: str, research_id: str = None, context_data: dict = None):
    """Save a message to the current conversation."""
    if not st.session_state.current_conversation_id:
        logger.warning("No active conversation to save message to")
        return None
    
    try:
        # Create message object for display immediately
        msg_display = {
            "id": str(hash(content))[:8],  # Temporary ID until API responds
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "research_id": research_id,
        }
        
        # Add to session state immediately (for UI continuity)
        st.session_state.conversation_messages.append(msg_display)
        logger.info(f"Added {role} message to session display. Total messages: {len(st.session_state.conversation_messages)}")
        
        # Now persist to database
        message = conversation_client.add_message(
            conversation_id=st.session_state.current_conversation_id,
            role=role,
            content=content,
            research_id=research_id,
            metadata=context_data,
        )
        
        if message:
            # Update with actual API response
            msg_display["id"] = str(message.get("id", msg_display["id"]))
            msg_display["created_at"] = message.get("created_at", msg_display["created_at"])
            logger.info(f"Saved {role} message to conversation {st.session_state.current_conversation_id}")
            return message
        else:
            logger.error(f"Failed to save {role} message - API returned None, but kept in display")
            return None
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return None


def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_client.delete_conversation(conversation_id):
        logger.info(f"Deleted conversation: {conversation_id}")
        if st.session_state.current_conversation_id == conversation_id:
            # Clear current conversation
            clear_session()
        return True
    return False


def rename_conversation(conversation_id: str, new_title: str):
    """Rename a conversation."""
    result = conversation_client.update_conversation(conversation_id, new_title)
    if result:
        if st.session_state.current_conversation_id == conversation_id:
            st.session_state.current_conversation_title = new_title
        logger.info(f"Renamed conversation to: {new_title}")
        return result
    return None

def get_recent_research(limit: int = 10):
    """Fetch recent research sessions from the database."""
    db = SessionLocal()
    try:
        return db.query(Research).order_by(Research.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.error(f"Error fetching research: {e}")
        return []
    finally:
        db.close()


def get_research_state_by_id(research_id: str):
    """Reconstruct a ResearchState object from the database for a given ID."""
    db = SessionLocal()
    try:
        res = db.query(Research).filter(Research.id == research_id).first()
        if not res:
            return None
        
        # Reconstruct state
        state = StateClass(
            query=res.query,
            search_queries=[],
            final_answer=res.final_answer,
            chat_history=res.chat_history or [],
            confidence_score=res.confidence_score,
            data_quality_score=res.data_quality_score,
            hallucination_flagged=res.hallucination_flagged,
            research_id=str(res.id),
            processed_urls=[],
            has_new_data=True,
            extracted_entities=[],
            conversation_summary=None,
            understood_intent=getattr(res, 'understood_intent', res.query),
            query_reasoning=getattr(res, 'query_reasoning', ""),
            active_topic=getattr(res, 'active_topic', ""),
            is_follow_up=getattr(res, 'is_follow_up', False),
            entities_resolved=getattr(res, 'entities_resolved', {}),
            conversational_knowledge=getattr(res, 'conversational_knowledge', {}) or {},
            scoped_entities=[],
            scope_context="",
        )
        
        # Load sources
        sources = db.query(Source).filter(Source.research_id == res.id).all()
        scored_sources = []
        for s in sources:
            scored_sources.append({
                "title": s.title,
                "url": s.url,
                "content": s.content,
                "overall_score": s.source_score,
                "relevance": s.relevance_score,
                "authority": s.authority_score,
                "freshness": s.recency_score,
                "domain_authority": s.authority_score,
                "content_freshness": s.recency_score
            })
            state.processed_urls.append(s.url)
            
        state.scored_sources = scored_sources
        return state
    except Exception as e:
        logger.error(f"Error getting research state {research_id}: {e}")
        return None
    finally:
        db.close()

@st.cache_data(show_spinner=False)
def get_pdf_report_bytes(research_id: str):
    """Cached PDF generation for a research ID."""
    state = get_research_state_by_id(research_id)
    if state:
        try:
            return generate_pdf_from_state(state)
        except Exception as e:
            logger.error(f"Error generating PDF for {research_id}: {e}")
            return None
    return None

def load_research_to_state(research_id: str, load_associated_convo: bool = True):
    """Load a specific research session into the current application state."""
    # Find and load the conversation this research belongs to if requested
    if load_associated_convo:
        db = SessionLocal()
        try:
            message = db.query(Message).filter(Message.research_id == research_id).first()
            if message:
                logger.info(f"Found associated conversation {message.conversation_id} for research {research_id}")
                convo = conversation_client.get_conversation(str(message.conversation_id))
                if convo:
                    st.session_state.current_conversation_id = str(message.conversation_id)
                    st.session_state.current_conversation_title = convo.get("title", "Chat")
                    st.session_state.conversation_messages = convo.get("messages", [])
        except Exception as e:
            logger.error(f"Error finding associated conversation: {e}")
        finally:
            db.close()
            
    state = get_research_state_by_id(research_id)
    if not state:
        return False
        
    st.session_state.research_state = state
    st.session_state.history = state.chat_history or []
    st.session_state.query = state.query
    st.session_state.current_research_id = research_id
    st.query_params["id"] = research_id
    return True

# Page config

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="search",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging capture
setup_logging_capture()

# Session state initialization

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "query": "",
        "research_state": None,
        "is_researching": False,
        "history": [], # List of {role: str, content: str}
        "last_final_answer": "",
        "current_query": "",
        "current_research_id": None,
        # Conversation variables
        "current_conversation_id": None,
        "current_conversation_title": "New Chat",
        "conversation_messages": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# Utility functions

def clear_session():
    """Reset current session for a new chat."""
    st.session_state.history = []
    st.session_state.research_state = None
    st.session_state.query = ""
    st.session_state.current_research_id = None
    st.session_state.conversation_messages = []
    st.session_state.current_conversation_id = None
    st.session_state.current_conversation_title = "New Chat"
    st.query_params.clear()

def get_process_logs() -> str:
    """Get all captured process logs as a single formatted string."""
    handler = get_streamlit_handler()
    logs = handler.get_logs()
    
    if not logs:
        return ""
    
    formatted_logs = []
    for log in logs:
        level = log["level"]
        timestamp = log["timestamp"]
        logger_name = log["logger"].split(".")[-1]
        message = log["message"]
        
        # Format: [12:34:56.789] INFO - logger_name - message
        line = f"[{timestamp}] {level:<7} - {logger_name:<15} - {message}"
        formatted_logs.append(line)
        
    return "\n".join(formatted_logs)


def render_logs_section(container):
    """Render real-time logs in a dedicated container."""
    logs_text = get_process_logs()
    if logs_text:
        container.code(logs_text, language="text")
    else:
        container.info("Initializing research process...")


def extract_sources_from_state(state: StateClass) -> List[Dict[str, Any]]:
    """Extract all sources from research state with complete details."""
    sources = []
    
    # Handle both raw search responses and scored sources
    if hasattr(state, 'scored_sources') and state.scored_sources:
        for idx, s in enumerate(state.scored_sources, 1):
            source = {
                "id": f"S{idx}",
                "query": "Scored Source",
                "title": s.get("title", "Untitled"),
                "url": s.get("url", ""),
                "score": s.get("overall_score", 0.0),
                "content_preview": s.get("content", "")[:200] + "...",
                "content_full": s.get("content", ""),
                "round": 1,
                "position": idx,
                "total_content_length": len(s.get("content", "")),
                "authority": s.get("authority") or s.get("domain_authority", 0.0),
                "freshness": s.get("freshness") or s.get("content_freshness", 0.0),
            }
            sources.append(source)
    else:
        for resp_idx, resp in enumerate(state.search_responses, 1):
            for result_idx, result in enumerate(resp.results, 1):
                source = {
                    "id": f"R{resp_idx}-{result_idx}",
                    "query": resp.query,
                    "title": result.title,
                    "url": result.url,
                    "score": result.score,
                    "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "content_full": result.content,
                    "round": resp_idx,
                    "position": result_idx,
                    "total_content_length": len(result.content),
                }
                sources.append(source)
    
    return sources


def display_sources_table(sources: List[Dict[str, Any]]):
    """Display sources in a comprehensive table."""
    if not sources:
        st.warning("No sources found in research")
        return
    
    st.markdown("### Sources Summary")
    
    # Create data for display
    table_data = []
    for src in sources:
        row = {
            "ID": src["id"],
            "Title": src["title"][:60],
            "Score": f"{src['score']:.1%}",
            "Chars": f"{src['total_content_length']:,}",
        }
        if "authority" in src:
            row["Auth"] = f"{src['authority']:.1%}"
        if "freshness" in src:
            row["Fresh"] = f"{src['freshness']:.1%}"
            
        table_data.append(row)
    
    st.dataframe(table_data, use_container_width=True, hide_index=True)


def display_source_details(sources: List[Dict[str, Any]]):
    """Display detailed information for each source."""
    st.markdown("### Source Details")
    
    for src in sources:
        with st.expander(f"Source {src['id']}: {src['title']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Source Information")
                st.markdown(f"**Query Used:** `{src['query']}`")
                st.markdown(f"**URL:** [{src['url']}]({src['url']})")
                st.markdown(f"**Title:** {src['title']}")
            
            with col2:
                st.markdown("#### Metrics")
                st.metric("Overall Score", f"{src['score']:.1%}")
                st.metric("Content Length", f"{src['total_content_length']:,} chars")
                if "round" in src:
                    st.metric("From Round", src['round'])
            
            st.markdown("#### Full Content")
            st.text_area(
                "Content",
                value=src['content_full'],
                height=250,
                disabled=True,
                key=f"source_content_{src['id']}"
            )


def run_research(query: str, is_follow_up: bool = False):
    """Execute research with real-time UI updates using a background thread."""
    st.session_state.is_researching = True
    
    # Clear logs
    handler = get_streamlit_handler()
    handler.clear_logs()
    
    try:
        cfg = get_agent_config()
        
        # Status and Log Containers
        status_container = st.empty()
        log_expander = st.expander("🛠️ Research Process (Live Terminal)", expanded=True)
        log_container = log_expander.empty()
        
        status_container.info(f"🚀 Initializing research for: {query}")
        
        # Shared result container for thread
        result_box = {"state": None, "error": None, "done": False}
        
        # Capture current state to pass to thread safely
        current_history = list(st.session_state.history)
        current_research_state = st.session_state.research_state
        
        def research_thread_target(history_copy, research_state_copy):
            try:
                graph = build_research_graph(cfg)
                
                # Update history for the thread's initial state
                temp_history = list(history_copy)
                temp_history.append({"role": "user", "content": query})
                
                if is_follow_up and research_state_copy:
                    initial_state = research_state_copy
                    initial_state.query = query
                    initial_state.chat_history = temp_history
                else:
                    initial_state = StateClass(query=query, chat_history=temp_history)
                
                final_state = graph.invoke(initial_state)
                
                if isinstance(final_state, dict):
                    final_state = StateClass(**{k: v for k, v in final_state.items() if k in StateClass.__dataclass_fields__})
                
                result_box["state"] = final_state
                result_box["done"] = True
            except Exception as thread_exc:
                logger.error(f"Error in research thread: {thread_exc}", exc_info=True)
                result_box["error"] = str(thread_exc)
                result_box["done"] = True

        # Start research in background
        research_thread = threading.Thread(
            target=research_thread_target, 
            args=(current_history, current_research_state)
        )
        research_thread.start()
        
        # Loop while thread is active to update UI
        start_time = time.time()
        while not result_box["done"]:
            # Update logs
            render_logs_section(log_container)
            
            # Update status message with timer
            elapsed = int(time.time() - start_time)
            status_container.info(f"🔍 Researching: {query}... ({elapsed}s)")
            
            time.sleep(1) # Refresh every second
            
        # Research complete
        if result_box["error"]:
            st.error(f"Research failed: {result_box['error']}")
            st.session_state.is_researching = False
            return None
            
        final_state = result_box["state"]
        status_container.success("✅ Research Complete")
        render_logs_section(log_container) # Final log update
        
        # Update session state
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.research_state = final_state
        st.session_state.last_final_answer = final_state.final_answer
        st.session_state.history.append({"role": "assistant", "content": final_state.final_answer})
        
        # Save to database
        if not st.session_state.current_conversation_id:
            create_new_conversation(title=query[:50], preserve_research_state=True)
        
        save_message_to_conversation(role="user", content=query, context_data={"type": "search_query"})
        
        rid = getattr(final_state, 'research_id', None)
        save_message_to_conversation(
            role="assistant",
            content=final_state.final_answer or "",
            research_id=str(rid) if rid else None,
            context_data={
                "type": "research_result",
                "confidence_score": final_state.confidence_score,
                "data_quality_score": final_state.data_quality_score,
                "sources_count": len(getattr(final_state, 'scored_sources', [])),
            }
        )
        
        final_state.chat_history = st.session_state.history
        st.session_state.is_researching = False
        
        return final_state
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.is_researching = False
        return None


def display_research_results(state: StateClass):
    """Display comprehensive research results."""
    if not state:
        st.warning("No research state available.")
        return

    if getattr(state, 'error', None):
        st.error(f"Error: {state.error}")
        return
    
    try:
        # Conversational Cognition (v2.3)
        if getattr(state, 'is_follow_up', False) and getattr(state, 'understood_intent', ''):
            with st.container(border=True):
                st.markdown("#### 🤖 Contextual Understanding")
                st.info(f"**Intent:** {state.understood_intent}")
                if getattr(state, 'query_reasoning', ''):
                    st.caption(f"**Reasoning:** {state.query_reasoning}")

        # 1. Summary Metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Confidence", f"{getattr(state, 'confidence_score', 0):.1%}")
        with cols[1]:
            st.metric("Data Quality", f"{getattr(state, 'data_quality_score', 0):.1%}")
        with cols[2]:
            st.metric("Iterations", getattr(state, 'iteration', 0))
        with cols[3]:
            st.metric("Sources", len(getattr(state, 'scored_sources', [])))

        # 2. Final answer
        st.markdown("## Synthesized Answer")
        final_answer = getattr(state, 'final_answer', None)
        if final_answer:
            st.markdown(final_answer)
        else:
            st.info("Synthesizing answer... please wait if research is still in progress.")

        # 3. Validation Results
        validation_results = getattr(state, 'validation_results', None)
        if validation_results and validation_results.get("results"):
            with st.expander("Validation & Quality Checks", expanded=False):
                for res in validation_results["results"]:
                    if res.get("passed"):
                        st.success(f"✅ {res.get('validation_type').title()}: {res.get('reason')}")
                    else:
                        st.warning(f"⚠️ {res.get('validation_type').title()}: {res.get('reason')}")

        # 4. Sources
        st.divider()
        sources = extract_sources_from_state(state)
        if sources:
            display_sources_table(sources)
            display_source_details(sources)
        else:
            st.info("No sources were captured in this research session.")

        # 5. Export Options
        st.divider()
        st.markdown("### 📥 Export Report")
        
        col1, col2, col3 = st.columns(3)
        
        # PDF Export
        with col1:
            try:
                pdf_bytes = generate_pdf_from_state(state)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                logger.error(f"Error generating PDF: {e}", exc_info=True)
                st.error("Could not generate PDF")
        
        # Markdown Export (Placeholder for future)
        with col2:
            st.button("📝 Export Markdown", disabled=True, use_container_width=True)
        
        # JSON Export (Placeholder for future)
        with col3:
            st.button("📊 Export JSON", disabled=True, use_container_width=True)

        # 6. Debug Info (Expander)
        with st.expander("Technical State (Debug)", expanded=False):
            st.json({
                "iteration": getattr(state, 'iteration', 0),
                "entities": getattr(state, 'extracted_entities', []),
                "queries": getattr(state, 'search_queries', []),
                "has_new_data": getattr(state, 'has_new_data', False),
                "info_gain": getattr(state, 'information_gain', 0.0)
            })
    except Exception as e:
        logger.error(f"Error in display_research_results: {e}", exc_info=True)
        st.error(f"Error displaying research results: {str(e)}")
        st.caption("(Check logs for more details)")



# Main application

def main():
    """Main Streamlit application."""
    
    # Auto-load most recent conversation ONLY if starting completely fresh
    # (no conversation ID and no messages in session)
    if (not st.session_state.current_conversation_id and 
        not st.session_state.conversation_messages and 
        not st.session_state.get("_auto_load_attempted", False)):
        st.session_state._auto_load_attempted = True
        conversations = get_all_conversations()
        if conversations:
            most_recent = conversations[0]  # Already sorted by recency
            logger.info(f"Auto-loading most recent conversation: {most_recent['id']}")
            load_conversation(str(most_recent["id"]))
            st.rerun()
    
    # Handle query params
    if "id" in st.query_params and st.session_state.current_research_id is None:
        research_id = st.query_params["id"]
        load_research_to_state(research_id)

    # Header
    st.markdown("# Deep Research Agent")
    st.markdown(
        "Complete autonomous research with real-time process tracking, "
        "source verification, and data completeness validation"
    )
    
    # Show current conversation header if one is loaded
    if st.session_state.current_conversation_id:
        st.markdown(f"## 💬 {st.session_state.current_conversation_title}")
        st.divider()
    
    # Conversation history
    if st.session_state.conversation_messages:
        for msg in st.session_state.conversation_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # If there's research associated with this message, show a button to load it
                if msg.get("research_id"):
                    rid = str(msg["research_id"])
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("🔍 View Report", key=f"view_res_{msg['id']}", use_container_width=True):
                            load_research_to_state(rid)
                            st.rerun()
                    with c2:
                        # Cached PDF generation for this specific turn
                        pdf_bytes = get_pdf_report_bytes(rid)
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_bytes,
                                file_name=f"research_report_{rid[:8]}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{msg['id']}",
                                use_container_width=True
                            )
    else:
        # Show a helpful starting message if no conversation is active
        if not st.session_state.current_conversation_id:
            st.info("💡 **Start by asking a question**")
        else:
            st.info("📝 **This conversation has no messages yet. Ask a question to get started!**")

    
    # Sidebar
    with st.sidebar:
        # Header with new chat and refresh buttons
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown("### 🔍 Research Agent")
        with col2:
            if st.button("➕", help="New Chat"):
                clear_session()
                st.rerun()
        with col3:
            if st.button("🔄", help="Refresh API Connection"):
                st.rerun()
        
        st.divider()
        
        # Conversations Section
        st.markdown("### 💬 Conversations")
        try:
            conversations = get_all_conversations()
            if conversations:
                for convo in conversations:
                    # Highlight active conversation
                    is_active = str(convo["id"]) == st.session_state.current_conversation_id
                    title = convo.get("title", "Untitled Chat")
                    if len(title) > 30:
                        title = title[:27] + "..."
                    
                    # Use columns for conversation entry and delete button
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        btn_label = f"▶ {title}" if is_active else title
                        if st.button(btn_label, key=f"convo_{convo['id']}", use_container_width=True):
                            load_conversation(str(convo["id"]))
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"del_{convo['id']}", help="Delete chat"):
                            if delete_conversation(str(convo["id"])):
                                st.rerun()
            else:
                st.caption("No conversations found.")
                if not conversation_client.check_health():
                    st.error("⚠️ API is unreachable")
                    st.info(f"Target: {conversation_client.base_url}")
        except Exception as e:
            st.error(f"API Error: {e}")

        st.divider()
        
        # Research Reports History (as an expander)
        with st.expander("📊 Recent Reports", expanded=False):
            recent = get_recent_research(15)
            if recent:
                for res in recent:
                    date_str = res.created_at.strftime("%y-%m-%d %H:%M")
                    is_current = str(res.id) == st.session_state.current_research_id
                    label = f"{res.query[:25]}..."
                    if is_current:
                        label = f"▶ {label}"
                    
                    if st.button(f"{label} ({date_str})", key=f"load_{res.id}", use_container_width=True):
                        # Load research and its parent conversation
                        if load_research_to_state(str(res.id), load_associated_convo=True):
                            st.rerun()
            else:
                st.caption("No reports yet.")

        st.divider()
        st.markdown("#### ⚙️ Configuration")

        cfg = get_agent_config()
        with st.container(border=True):
            st.write(f"**LLM:** {cfg.llm_provider.value}")
            st.write(f"**Search:** {cfg.search_provider.value}")
            st.write(f"**Depth:** {cfg.research_depth.value}")
            st.write(f"**Scraper:** {'ON' if cfg.enable_scraper else 'OFF'}")

    # Current research results
    # Only show the full research report for the LATEST turn
    if st.session_state.research_state:
        logger.debug(f"Research state found. is_researching={st.session_state.is_researching}")
        if not st.session_state.is_researching:
            st.divider()
            st.markdown("### Latest Research Report")
            logger.debug("Rendering research results...")
            try:
                display_research_results(st.session_state.research_state)
            except Exception as e:
                logger.error(f"Error displaying research results: {e}", exc_info=True)
                st.error(f"Error displaying research results: {str(e)}")
                st.info("Try refreshing the page or selecting the conversation again.")
            
            # After a run, we should have a research_id
            if hasattr(st.session_state.research_state, 'research_id') and st.session_state.research_state.research_id:
                rid = str(st.session_state.research_state.research_id)
                if st.session_state.current_research_id != rid:
                    st.session_state.current_research_id = rid
                    st.query_params["id"] = rid
        else:
            st.info("Research in progress... 🔍")

    # Input section
    if not st.session_state.is_researching:
        prompt = st.chat_input("Ask a new question or follow-up...")
        if prompt:
            logger.info(f"New prompt received: {prompt[:100]}...")
            logger.info(f"Current messages count: {len(st.session_state.conversation_messages)}")
            # Determine if it's a follow-up
            is_follow_up = st.session_state.research_state is not None
            st.session_state.query = prompt
            run_research(prompt, is_follow_up=is_follow_up)
            logger.info(f"After research - messages count: {len(st.session_state.conversation_messages)}")
            st.rerun()
    else:
        st.info("Agent is researching... please wait. Check the log expander above for live process tracking.")

    # Footer
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Deep Research Agent")


if __name__ == "__main__":
    main()
