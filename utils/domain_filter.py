"""Utilities for URL normalization and domain filtering."""
import re
from urllib.parse import urlparse, urlunparse

# Domains that are often low-quality for research or too large to scrape effectively
EXCLUDED_DOMAINS = {
    "reddit.com",
    "wikipedia.org",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "vimeo.com",
    "amazon.com",
    "ebay.com",
    "quora.com",
    "pinterest.com",
    "tumblr.com",
    "medium.com", # Blogs
    "blogspot.com",
    "wordpress.com",
}

def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication.
    - Lowercase scheme and host
    - Remove default ports
    - Remove trailing slash
    - Remove common tracking parameters (utm_*)
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        
        # Remove default ports
        if scheme == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif scheme == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]
            
        path = parsed.path
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
            
        # Remove common tracking parameters
        query = parsed.query
        if query:
            params = query.split("&")
            params = [p for p in params if not p.startswith("utm_") and not p.startswith("ref_")]
            query = "&".join(params)
            
        return urlunparse((scheme, netloc, path, parsed.params, query, ""))
    except Exception:
        return url.strip().lower().rstrip("/")

def is_excluded_domain(url: str) -> bool:
    """Check if a URL belongs to an excluded domain."""
    if not url:
        return False
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return any(domain == ex or domain.endswith("." + ex) for ex in EXCLUDED_DOMAINS)
    except Exception:
        return False

def should_skip_scraping(url: str) -> bool:
    """
    Determine if a URL should be skipped for scraping.
    Skips excluded domains and non-HTML files.
    """
    if not url:
        return True
    
    if is_excluded_domain(url):
        return True
        
    # Skip common non-HTML file extensions
    skip_extensions = {
        ".pdf", ".zip", ".exe", ".png", ".jpg", ".jpeg", 
        ".gif", ".mp4", ".mp3", ".wav", ".csv", ".xlsx", 
        ".docx", ".pptx", ".gz", ".tar", ".dmg", ".iso"
    }
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in skip_extensions):
        return True
        
    return False
