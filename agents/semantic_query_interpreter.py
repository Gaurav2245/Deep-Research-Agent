"""
Semantic Query Interpreter

Transforms malformed conversational queries into structured retrievable intent.

Problem: "tell me strike rate of above mentioned strike rate" 
→ Entity extractor finds nothing
→ Query planner finds nothing
→ Collapse

Solution: Use context + intent parsing + semantic rewriting to infer:
"Show strike rates of previously mentioned batsmen (Klaasen, Sharma, etc.)"

This is query interpretation, not memory retrieval.
Memory was already loaded. The problem is semantic reconstruction of malformed input.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re
from enum import Enum

from agents.state import ResearchState
from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationalIntent(Enum):
    """Types of conversational intent"""
    INFORMATION_SEEKING = "information_seeking"
    COMPARISON = "comparison"
    RANKING = "ranking"
    CLARIFICATION = "clarification"
    AGGREGATION = "aggregation"  # "all batsmen", "every player"
    DETAIL = "detail"  # "tell me about", "explain"
    REPETITION = "repetition"  # "say it again", "repeat"
    UNKNOWN = "unknown"


@dataclass
class ResolvedReference:
    """A resolved pronoun or shorthand reference"""
    original_text: str  # "above mentioned"
    resolved_to: List[str]  # ["Klaasen", "Sharma", "Abhishek"]
    reference_type: str  # "pronoun", "ellipsis", "shorthand", "context_reference"
    confidence: float


@dataclass
class SemanticInterpretation:
    """Semantic interpretation of a query"""
    original_query: str
    normalized_query: str  # Rewritten for clarity
    inferred_intent: ConversationalIntent
    intent_confidence: float
    
    # Context used
    context_entities: List[str]  # Entities brought from context
    context_metrics: List[str]  # Metrics from context ("strike rate", "runs")
    active_topics: List[str]  # Topics active in conversation
    
    # Resolved elements
    resolved_references: List[ResolvedReference]
    
    # Reconstructed meaning
    query_reconstruction: str  # What we think user meant
    reconstruction_confidence: float
    
    # Actionable output
    search_entities: List[str]  # Entities to search for
    search_metrics: List[str]  # Metrics to search for
    search_context: str  # Additional context for planner
    
    # Confidence in interpretation
    overall_confidence: float
    warnings: List[str] = field(default_factory=list)
    
    def should_use_interpretation(self, threshold: float = 0.5) -> bool:
        """Should we use this interpretation or fall back?"""
        return self.overall_confidence >= threshold


class SemanticQueryInterpreter:
    """
    Interprets semantic meaning of queries using conversational context.
    
    Handles:
    - Malformed queries: "strike rate of above mentioned strike rate"
    - Ellipsis: "what about him"
    - Shorthand: "same player stats"
    - Omitted entities: "ranking" (ranking of what? Use context)
    - Ambiguous intent: "tell me" (what aspect?)
    
    Usage:
        interpreter = SemanticQueryInterpreter()
        interpretation = interpreter.interpret(
            query="tell me strike rate of above mentioned strike rate",
            context_entities=["Klaasen", "Sharma", ...],
            context_metrics=["strike rate", "runs"],
            active_topics=["IPL 2026"],
            conversation_state=state
        )
        
        if interpretation.should_use_interpretation():
            # Use interpretation.search_entities, search_metrics, normalized_query
    """
    
    def interpret(
        self,
        query: str,
        context_entities: Optional[List[str]] = None,
        context_metrics: Optional[List[str]] = None,
        active_topics: Optional[List[str]] = None,
        conversation_state: Optional[Any] = None,
        previous_answer: Optional[str] = None
    ) -> SemanticInterpretation:
        """
        Interpret semantic meaning of query.
        
        Args:
            query: Raw user query (may be malformed)
            context_entities: Entities from prior conversation
            context_metrics: Metrics mentioned before
            active_topics: Topics being discussed
            conversation_state: Full conversation context
            previous_answer: Last answer given
        
        Returns:
            SemanticInterpretation with reconstruction
        """
        context_entities = context_entities or []
        context_metrics = context_metrics or []
        active_topics = active_topics or []
        
        logger.info(f"Interpreting: {query[:60]}...")
        
        # 1. Parse intent
        intent = self._parse_intent(query)
        intent_confidence = self._calculate_intent_confidence(query, intent)
        
        # 2. Resolve references
        resolved = self._resolve_references(query, context_entities)
        
        # 3. Extract mentioned metrics
        mentioned_metrics = self._extract_metrics(query, context_metrics)
        
        # 4. Reconstruct meaning
        reconstruction, rec_confidence = self._reconstruct_meaning(
            query=query,
            intent=intent,
            context_entities=context_entities,
            resolved_references=resolved,
            mentioned_metrics=mentioned_metrics,
            context_metrics=context_metrics,
            active_topics=active_topics
        )
        
        # 5. Extract search entities (resolved references + context)
        search_entities = [r.resolved_to for r in resolved]
        search_entities = [item for sublist in search_entities for item in sublist]
        
        if not search_entities and context_entities:
            # If no entities resolved, use context entities for aggregation intents
            if intent in [ConversationalIntent.AGGREGATION, ConversationalIntent.RANKING]:
                search_entities = context_entities
        
        # 6. Normalize query for planner
        normalized = self._normalize_query(
            query,
            reconstruction,
            search_entities,
            mentioned_metrics or context_metrics,
            active_topics
        )
        
        # 7. Calculate overall confidence
        overall_confidence = (intent_confidence + rec_confidence) / 2
        
        # 8. Generate warnings
        warnings = []
        if intent_confidence < 0.5:
            warnings.append("Low confidence in intent detection")
        if rec_confidence < 0.5:
            warnings.append("Query reconstruction uncertain")
        if not search_entities:
            warnings.append("No entities found - using context")
        if not mentioned_metrics:
            warnings.append("No metrics mentioned - using context")
        
        interpretation = SemanticInterpretation(
            original_query=query,
            normalized_query=normalized,
            inferred_intent=intent,
            intent_confidence=intent_confidence,
            context_entities=context_entities,
            context_metrics=context_metrics,
            active_topics=active_topics,
            resolved_references=resolved,
            query_reconstruction=reconstruction,
            reconstruction_confidence=rec_confidence,
            search_entities=search_entities or context_entities,
            search_metrics=mentioned_metrics or context_metrics,
            search_context=active_topics[0] if active_topics else "",
            overall_confidence=overall_confidence,
            warnings=warnings
        )
        
        logger.info(
            f"Interpretation: intent={intent.value} | confidence={overall_confidence:.2f} | "
            f"entities={len(interpretation.search_entities)} | "
            f"reconstruction={reconstruction[:50]}..."
        )
        
        return interpretation
    
    def _parse_intent(self, query: str) -> ConversationalIntent:
        """Detect intent from query text"""
        query_lower = query.lower()
        
        # Comparison intent
        if any(w in query_lower for w in ['compare', 'vs', 'versus', 'difference', 'vs.', 'better']):
            return ConversationalIntent.COMPARISON
        
        # Ranking intent
        if any(w in query_lower for w in ['top', 'best', 'highest', 'lowest', 'rank', 'ranking', 'most']):
            return ConversationalIntent.RANKING
        
        # Aggregation intent ("all", "every", "list of")
        if any(w in query_lower for w in ['all', 'every', 'each', 'list', 'tell me', 'show']):
            return ConversationalIntent.AGGREGATION
        
        # Detail intent ("tell me about", "explain")
        if any(w in query_lower for w in ['tell me', 'explain', 'about', 'detail', 'more']):
            return ConversationalIntent.DETAIL
        
        # Clarification intent
        if any(w in query_lower for w in ['what', 'which', 'who', 'where', 'when', 'how']):
            return ConversationalIntent.CLARIFICATION
        
        # Repetition intent
        if any(w in query_lower for w in ['again', 'repeat', 'say', 'same', 'another']):
            return ConversationalIntent.REPETITION
        
        # Default
        return ConversationalIntent.INFORMATION_SEEKING
    
    def _calculate_intent_confidence(self, query: str, intent: ConversationalIntent) -> float:
        """Confidence in detected intent"""
        if intent == ConversationalIntent.UNKNOWN:
            return 0.2
        
        # Clear intent signals boost confidence
        clear_signals = {
            ConversationalIntent.COMPARISON: ['compare', 'vs', 'difference'],
            ConversationalIntent.RANKING: ['top', 'best', 'highest'],
            ConversationalIntent.AGGREGATION: ['all', 'every', 'list'],
            ConversationalIntent.DETAIL: ['tell me', 'explain'],
        }
        
        query_lower = query.lower()
        if intent in clear_signals:
            signal_count = sum(1 for s in clear_signals[intent] if s in query_lower)
            return min(0.5 + (signal_count * 0.2), 0.95)
        
        # Default confidence
        return 0.7
    
    def _resolve_references(
        self,
        query: str,
        context_entities: List[str]
    ) -> List[ResolvedReference]:
        """Resolve pronouns and context references"""
        resolved = []
        query_lower = query.lower()
        
        # "above mentioned" pattern
        if "above mentioned" in query_lower or "mentioned above" in query_lower:
            resolved.append(ResolvedReference(
                original_text="above mentioned",
                resolved_to=context_entities,
                reference_type="context_reference",
                confidence=0.9
            ))
        
        # "previously mentioned" / "before"
        if any(p in query_lower for p in ["previously mentioned", "mentioned before", "earlier"]):
            resolved.append(ResolvedReference(
                original_text="previously mentioned",
                resolved_to=context_entities,
                reference_type="context_reference",
                confidence=0.85
            ))
        
        # Pronouns with context
        if context_entities:
            pronouns = {
                "his": context_entities[0] if context_entities else None,
                "her": context_entities[0] if context_entities else None,
                "they": context_entities,
                "them": context_entities,
                "their": context_entities,
            }
            
            for pronoun, entity in pronouns.items():
                if pronoun in query_lower and entity:
                    resolved.append(ResolvedReference(
                        original_text=pronoun,
                        resolved_to=[entity] if isinstance(entity, str) else entity,
                        reference_type="pronoun",
                        confidence=0.8
                    ))
        
        # "same" reference
        if "same" in query_lower and context_entities:
            resolved.append(ResolvedReference(
                original_text="same",
                resolved_to=context_entities[:1],
                reference_type="ellipsis",
                confidence=0.7
            ))
        
        return resolved
    
    def _extract_metrics(self, query: str, context_metrics: List[str]) -> List[str]:
        """Extract metrics mentioned in query"""
        query_lower = query.lower()
        
        # Common metrics
        metric_keywords = {
            'strike rate': ['strike', 'sr', 'strike rate', 'strike-rate'],
            'runs': ['runs', 'scored', 'total runs'],
            'average': ['average', 'avg', 'batting average'],
            'wickets': ['wickets', 'wkts'],
            'centuries': ['century', 'hundred', 'centuries', 'tons'],
            'fours': ['fours', '4s'],
            'sixes': ['sixes', '6s'],
        }
        
        extracted = []
        
        for metric, keywords in metric_keywords.items():
            if any(kw in query_lower for kw in keywords):
                extracted.append(metric)
        
        # Also include context metrics if query is vague
        if not extracted and context_metrics:
            extracted.extend(context_metrics)
        
        return extracted
    
    def _reconstruct_meaning(
        self,
        query: str,
        intent: ConversationalIntent,
        context_entities: List[str],
        resolved_references: List[ResolvedReference],
        mentioned_metrics: List[str],
        context_metrics: List[str],
        active_topics: List[str]
    ) -> tuple[str, float]:
        """Reconstruct meaning of query"""
        
        # Get entities that will be searched
        all_entities = set()
        for ref in resolved_references:
            all_entities.update(ref.resolved_to)
        
        if not all_entities and context_entities:
            all_entities = set(context_entities)
        
        # Get metrics
        metrics = mentioned_metrics or context_metrics or []
        
        # Build reconstruction based on intent
        if intent == ConversationalIntent.AGGREGATION:
            if all_entities:
                entities_str = ", ".join(list(all_entities)[:5])
                if metrics:
                    reconstruction = f"Show {', '.join(metrics)} for {entities_str}"
                else:
                    reconstruction = f"Show statistics for {entities_str}"
                confidence = 0.85
            else:
                reconstruction = "List all available data"
                confidence = 0.5
        
        elif intent == ConversationalIntent.DETAIL:
            if all_entities:
                entity = list(all_entities)[0]
                if metrics:
                    reconstruction = f"Detailed information about {entity}'s {', '.join(metrics)}"
                else:
                    reconstruction = f"Detailed information about {entity}"
                confidence = 0.8
            else:
                reconstruction = query  # Keep original
                confidence = 0.4
        
        elif intent == ConversationalIntent.COMPARISON:
            entities_list = list(all_entities) if all_entities else context_entities
            if len(entities_list) >= 2:
                reconstruction = f"Compare {entities_list[0]} and {entities_list[1]}"
                if metrics:
                    reconstruction += f" on {', '.join(metrics)}"
                confidence = 0.8
            else:
                reconstruction = query
                confidence = 0.5
        
        elif intent == ConversationalIntent.RANKING:
            if metrics:
                reconstruction = f"Rank batsmen by {', '.join(metrics)}"
                confidence = 0.8
            else:
                reconstruction = "Ranking of batsmen"
                confidence = 0.6
        
        else:
            reconstruction = query
            confidence = 0.5
        
        return reconstruction, confidence
    
    def _normalize_query(
        self,
        original: str,
        reconstruction: str,
        search_entities: List[str],
        search_metrics: List[str],
        active_topics: List[str]
    ) -> str:
        """Normalize query for planner"""
        
        # If reconstruction is very different, use it
        if reconstruction != original:
            # Add context
            context_str = ""
            if active_topics:
                context_str = f" in {active_topics[0]}"
            
            normalized = reconstruction + context_str
        else:
            normalized = original
        
        return normalized


def make_semantic_query_interpreter_node():
    """
    Create node that interprets semantic meaning of queries.
    
    Placement in graph: After state reconstruction, BEFORE memory retriever
    
    This ensures that malformed queries are semantically reconstructed
    BEFORE we attempt to retrieve from memory or plan retrieval.
    """
    def semantic_query_interpreter(state: ResearchState) -> ResearchState:
        interpreter = SemanticQueryInterpreter()
        
        # Get context
        context_entities = []
        context_metrics = []
        active_topics = []
        
        if hasattr(state, 'prior_entities'):
            context_entities = [e.name if hasattr(e, 'name') else e for e in state.prior_entities]
        
        if hasattr(state, 'conversation_topics'):
            active_topics = state.conversation_topics
        
        # Interpret query
        interpretation = interpreter.interpret(
            query=state.query,
            context_entities=context_entities,
            context_metrics=context_metrics,
            active_topics=active_topics,
            conversation_state=state,
            previous_answer=getattr(state, 'last_answer_text', None)
        )
        
        # Store interpretation
        state.semantic_interpretation = interpretation
        
        # Update state based on interpretation
        if interpretation.should_use_interpretation(threshold=0.4):
            # Use normalized query and reconstructed entities
            state.query_normalized = interpretation.normalized_query
            state.query_reconstruction = interpretation.query_reconstruction
            state.inferred_intent = interpretation.inferred_intent
            state.search_entities = interpretation.search_entities
            state.search_metrics = interpretation.search_metrics
            
            logger.info(
                f"Interpretation accepted | intent={interpretation.inferred_intent.value} | "
                f"entities={len(state.search_entities)} | metrics={len(state.search_metrics)}"
            )
        else:
            logger.warning(
                f"Low confidence interpretation ({interpretation.overall_confidence:.2f}), "
                f"using original query"
            )
        
        return state
    
    return semantic_query_interpreter
