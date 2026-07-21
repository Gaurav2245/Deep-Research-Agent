"""Source scoring, filtering, and selection system."""
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse

from utils.logger import get_logger

logger = get_logger(__name__)


class SourceScorer:
    """Score sources based on reliability, relevance, and recency."""

    # HIGH AUTHORITY: 1.0
    HIGH_AUTHORITY_DOMAINS = {
        "gov.in", "nic.in", "gov.uk", "sec.gov", "fed.gov",
        "nseindia.com", "bseindia.com", "rbi.org.in", "bloomberg.com", "reuters.com", "ft.com",
        "bbc.com", "nytimes.com", "theguardian.com", "thehindu.com", "ndtv.com",
        "arxiv.org", "nature.com", "pubmed.ncbi.nlm.nih.gov",
        "github.com", "stackoverflow.com", "developer.mozilla.org",
        # Sports Authority
        "espncricinfo.com", "espn.com", "foxsports.com", "goal.com", "soccerway.com",
        "atptour.com", "wtatennis.com", "nba.com", "fifa.com", "olympics.com"
    }

    @staticmethod
    def get_canonical_url(url: str) -> str:
        """
        Normalize URL to a canonical form for deduplication.
        Removes query params, fragments, and trailing slashes.
        """
        try:
            parsed = urlparse(url)
            # Remove www. and lowercase netloc
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            
            # Clean path: remove trailing slash
            path = parsed.path.rstrip("/")
            
            # Reconstruct without query/fragment
            return f"{netloc}{path}"
        except Exception:
            return url.lower().strip().rstrip("/")

    # MEDIUM AUTHORITY: 0.7
    MEDIUM_AUTHORITY_DOMAINS = {
        "medium.com", "substack.com", "forbes.com", "businessinsider.com", "cnbc.com"
    }

    # LOW AUTHORITY: 0.3
    LOW_AUTHORITY_DOMAINS = {
        "blogspot.com", "wordpress.com", "weebly.com"
    }

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL, normalized."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {e}")
            return ""

    @classmethod
    def score_domain_authority(cls, url: str) -> float:
        """
        Score domain authority based on the specified trust system.
        
        Returns: 0.0-1.0 score
        """
        domain = cls.extract_domain(url)
        score = 0.5  # Base score for unknown domains
        
        if domain in cls.HIGH_AUTHORITY_DOMAINS:
            score = 1.0
        elif domain in cls.MEDIUM_AUTHORITY_DOMAINS:
            score = 0.7
        elif domain in cls.LOW_AUTHORITY_DOMAINS:
            score = 0.3
        
        # Additional rules
        # Wikipedia penalized
        if "wikipedia.org" in domain:
            score = 0.4
            
        # Blogs penalized (generic check if not in low authority list)
        if "blog" in domain and score > 0.4:
            score = 0.4
            
        # "official", "government", "ministry" keywords boost trust
        boost_keywords = ["official", "government", "ministry"]
        if any(kw in url.lower() for kw in boost_keywords):
            score = min(score + 0.2, 1.0)
            
        return score

    @staticmethod
    def score_content_freshness(content_date: str | None = None) -> float:
        """
        Score content freshness based on publication date using exponential decay.
        """
        if not content_date:
            return 0.5  # Neutral score for unknown date
        
        try:
            date_str = content_date.strip()
            # Clean up common web date artifacts
            date_str = re.sub(r'^(?:published|updated|posted|on|dated)\s*(?::|on)?\s*', '', date_str, flags=re.IGNORECASE)
            
            pub_date = None
            # Try a wide range of formats
            formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
                "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                "%Y/%m/%d", "%d/%m/%Y",
                "%a, %d %b %Y %H:%M:%S %Z", # RFC 1123 / GMT
                "%a, %d %b %Y %H:%M:%S",
                "%Y" # Year only fallback
            ]
            
            for fmt in formats:
                try:
                    pub_date = datetime.strptime(date_str, fmt)
                    # If we only got a year, we don't break yet if we find a better format
                    if fmt != "%Y":
                        break
                except ValueError:
                    continue
            
            # Handle ISO formats and timestamps
            if not pub_date or date_str.count("-") >= 2 or "T" in date_str:
                try:
                    # Try isoformat but only the date part
                    clean_iso = date_str.split('T')[0].split(' ')[0]
                    pub_date = datetime.fromisoformat(clean_iso)
                except Exception:
                    pass
            
            if not pub_date:
                logger.debug(f"[SourceScorer] Failed to parse date string: '{content_date}'")
                return 0.5

            # Calculate days old
            now = datetime.utcnow()
            days_old = (now - pub_date).days
            
            # Future dates (edge case) treated as today
            if days_old < 0:
                days_old = 0
            
            # Exponential decay: score = exp(-lambda * days_old)
            # lambda = 0.005 means ~50% score at 138 days
            decay_constant = 0.005
            score = math.exp(-decay_constant * days_old)
            
            # If it was a Year Only match, we apply a small penalty because we don't know the exact day
            if len(date_str) == 4 and date_str.isdigit():
                score *= 0.9 # 10% uncertainty penalty for year-only
            
            return max(score, 0.1)
        except Exception as e:
            logger.warning(f"Error parsing date '{content_date}': {e}")
            return 0.5

    @staticmethod
    def score_content_quality(content: str, title: Optional[str] = None) -> float:
        """
        Score content quality based on length, tables, sentence count, paywall detection, and title.
        """
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length (up to 0.3)
        length_score = min(len(content) / 3000, 1.0) * 0.3
        score += length_score
        
        # Tables presence (up to 0.2)
        has_tables = "<table>" in content or "| --- |" in content or "<thead>" in content
        if has_tables:
            score += 0.2
            
        # Sentence count (up to 0.3)
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if len(s.strip()) > 5])
        sentence_score = min(sentence_count / 30, 1.0) * 0.3
        score += sentence_score
        
        # Title presence (up to 0.1)
        if title and len(title.strip()) > 5:
            score += 0.1
        
        # Paywall detection (penalty up to -0.5)
        paywall_keywords = ["subscribe to read", "paywall", "sign in to continue", "purchase a subscription", "exclusive content for"]
        if any(kw in content.lower() for kw in paywall_keywords):
            score = max(score - 0.5, 0.0)
            
        # Short content penalty (if already not covered by length/sentence)
        if len(content) < 200:
            score *= 0.5
            
        return min(score, 1.0)

    @classmethod
    def calculate_source_score(
        cls,
        url: str,
        raw_search_score: float = 0.5,
        content: str = "",
        query: str = "",
        content_date: str = None,
        title: str = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive source score based on user specifications.
        
        Weights:
        - relevance: 0.35
        - freshness: 0.35
        - authority: 0.20
        - content_quality: 0.10
        """
        authority_score = cls.score_domain_authority(url)
        freshness_score = cls.score_content_freshness(content_date)
        quality_score = cls.score_content_quality(content, title)
        
        # Relevance is raw_search_score (no semantic reranking)
        relevance_score = raw_search_score
        
        # Weighted average
        overall = (
            relevance_score * 0.35 +
            freshness_score * 0.35 +
            authority_score * 0.20 +
            quality_score * 0.10
        )
        
        return {
            "relevance": relevance_score,
            "freshness": freshness_score,
            "authority": authority_score,
            "content_quality": quality_score,
            "overall_score": overall,
        }


class SourceFilter:
    """Filter and select best sources based on scoring."""

    @staticmethod
    def filter_by_score(
        sources: List[Dict],
        min_score: float = 0.4,
        exclude_urls: set = None
    ) -> List[Dict]:
        """Filter sources by minimum score."""
        exclude_urls = exclude_urls or set()
        return [
            s for s in sources
            if s.get("overall_score", 0) >= min_score and s.get("url") not in exclude_urls
        ]

    @staticmethod
    def filter_by_domain_diversity(
        sources: List[Dict],
        max_per_domain: int = 3
    ) -> List[Dict]:
        """
        Filter to ensure diversity - not too many from same domain.
        """
        domain_counts = {}
        filtered = []
        
        for source in sources:
            domain = SourceScorer.extract_domain(source.get("url", ""))
            if domain_counts.get(domain, 0) < max_per_domain:
                filtered.append(source)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return filtered

    @staticmethod
    def select_best_sources(
        sources: List[Dict],
        count: int = 15,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Select best sources:
        1. Canonical URL deduplication
        2. Filter by minimum score
        3. Sort by score (descending)
        4. Strict domain diversity (max 2 per domain)
        5. Return top N
        """
        # 1. Canonical URL deduplication (prevent similar URLs from same site)
        seen_canonical = set()
        unique_sources = []
        for s in sources:
            canon = SourceScorer.get_canonical_url(s.get("url", ""))
            if canon not in seen_canonical:
                unique_sources.append(s)
                seen_canonical.add(canon)
        
        # 2. Filter low-scoring sources
        filtered = SourceFilter.filter_by_score(unique_sources, min_score)
        
        # 3. Sort by overall score
        filtered.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # 4. Strict domain diversity (max 2 per domain for higher information diversity)
        diverse = SourceFilter.filter_by_domain_diversity(filtered, max_per_domain=2)
        
        # 5. Return top N
        return diverse[:count]
