"""
utils/table_extractor.py
Standalone HTML-table → Markdown/DataFrame converter.
Used by nodes and tools that handle HTML content.
"""
from __future__ import annotations

import re
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def html_tables_to_markdown(html: str, max_tables: int = 10) -> List[str]:
    """
    Extract all <table> elements from an HTML string and convert each to
    a Markdown table.

    Parameters
    ----------
    html:       Raw HTML string.
    max_tables: Maximum number of tables to extract.

    Returns
    -------
    List of markdown-formatted table strings (one per <table> element).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError(
            "beautifulsoup4 is required for table extraction. "
            "Install it with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")[:max_tables]
    logger.debug("Found %d table(s) in HTML", len(tables))

    results: List[str] = []
    for idx, table in enumerate(tables, start=1):
        md = _table_element_to_markdown(table)
        if md:
            results.append(md)
            logger.debug("Table %d: %d rows extracted", idx, md.count("\n"))

    return results


def _table_element_to_markdown(table) -> Optional[str]:
    """Convert a single BeautifulSoup <table> element to markdown."""
    rows_data: List[List[str]] = []

    # Honour thead / tbody / tfoot if present, fall back to all <tr>
    sections = table.find_all(["thead", "tbody", "tfoot"]) or [table]
    for section in sections:
        for row in section.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if cells:
                rows_data.append([_cell_text(c) for c in cells])

    if not rows_data:
        return None

    # Normalise column count
    max_cols = max(len(r) for r in rows_data)
    rows_data = [r + [""] * (max_cols - len(r)) for r in rows_data]

    header = "| " + " | ".join(rows_data[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in rows_data[1:]]

    return "\n".join([header, separator] + body_lines)


def _cell_text(cell) -> str:
    text = cell.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


# ── DataFrame conversion (optional, requires pandas) ─────────────────────────

def html_tables_to_dataframes(html: str):
    """
    Parse all tables from HTML into pandas DataFrames.
    Requires pandas to be installed.

    Returns
    -------
    List[pd.DataFrame]
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required. Install with: pip install pandas")

    try:
        dfs = pd.read_html(html)
        logger.debug("pandas parsed %d table(s)", len(dfs))
        return dfs
    except ValueError:
        logger.debug("No tables found by pandas")
        return []
