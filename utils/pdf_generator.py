"""PDF report generation for research results with structured data rendering."""
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas


class ResearchReportDTO:
    """
    Structured data transfer object for research reports.
    
    Separates data from presentation, enabling proper formatting.
    """
    
    def __init__(
        self,
        query: str,
        understood_intent: str = "",
        summary: str = "",
        sections: List[Dict[str, Any]] = None,
        tables: List[Dict[str, Any]] = None,
        key_findings: List[str] = None,
        confidence_score: float = 0.0,
        data_quality_score: float = 0.0,
        sources: List[Dict[str, Any]] = None,
        validation_results: Dict[str, Any] = None,
        hallucination_flagged: bool = False,
        iteration_count: int = 0,
    ):
        self.query = query
        self.understood_intent = understood_intent or query
        self.summary = summary
        self.sections = sections or []  # List of {title, content, type} sections
        self.tables = tables or []  # List of {title, headers, rows} structured tables
        self.key_findings = key_findings or []
        self.confidence_score = confidence_score
        self.data_quality_score = data_quality_score
        self.sources = sources or []
        self.validation_results = validation_results or {}
        self.hallucination_flagged = hallucination_flagged
        self.iteration_count = iteration_count
        self.generated_at = datetime.now()


class MarkdownTableParser:
    """Parse markdown tables into structured data."""
    
    @staticmethod
    def parse_markdown_table(markdown_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract first markdown table from text.
        
        Returns: {headers: [...], rows: [[...], [...]]}
        """
        # Pattern: | ... | ... |
        lines = markdown_text.split('\n')
        tables = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this is a table header line
            if line.startswith('|') and '|' in line:
                headers = [cell.strip() for cell in line.split('|')[1:-1]]
                
                # Check for separator line
                if i + 1 < len(lines):
                    sep_line = lines[i + 1].strip()
                    if sep_line.startswith('|') and all('-' in cell for cell in sep_line.split('|')[1:-1]):
                        # This is a valid table
                        rows = []
                        j = i + 2
                        
                        while j < len(lines):
                            row_line = lines[j].strip()
                            if row_line.startswith('|') and '|' in row_line:
                                cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                                if len(cells) == len(headers):
                                    rows.append(cells)
                                    j += 1
                                else:
                                    break
                            else:
                                break
                        
                        if rows:
                            tables.append({
                                'headers': headers,
                                'rows': rows,
                                'start': i,
                                'end': j
                            })
                            i = j
                            continue
            
            i += 1
        
        return tables[0] if tables else None
    
    @staticmethod
    def extract_all_tables(markdown_text: str) -> List[Dict[str, Any]]:
        """Extract all markdown tables from text."""
        tables = []
        lines = markdown_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('|') and '|' in line:
                headers = [cell.strip() for cell in line.split('|')[1:-1]]
                
                if i + 1 < len(lines):
                    sep_line = lines[i + 1].strip()
                    if sep_line.startswith('|') and all('-' in cell for cell in sep_line.split('|')[1:-1]):
                        rows = []
                        j = i + 2
                        
                        while j < len(lines):
                            row_line = lines[j].strip()
                            if row_line.startswith('|') and '|' in row_line:
                                cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                                if len(cells) == len(headers):
                                    rows.append(cells)
                                    j += 1
                                else:
                                    break
                            else:
                                break
                        
                        if rows:
                            tables.append({
                                'headers': headers,
                                'rows': rows
                            })
                            i = j
                            continue
            
            i += 1
        
        return tables


def generate_pdf_report(report: ResearchReportDTO) -> bytes:
    """
    Generate professional PDF from structured research report.
    
    Uses proper data structures, not text blobs.
    Returns bytes for direct Streamlit download.
    """
    pdf_buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    
    # Define custom styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
    )
    
    subsection_heading = ParagraphStyle(
        'SubsectionHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        spaceBefore=8,
        fontName='Helvetica-Bold',
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    
    finding_style = ParagraphStyle(
        'Finding',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6,
        leftIndent=0.25*inch,
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    
    elements = []
    
    # ===== 1. COVER PAGE / TITLE =====
    elements.append(Paragraph("Deep Research Agent Report", title_style))
    elements.append(Spacer(1, 6))
    
    # Use understood_intent for title display if it's significantly different/better
    display_query = report.understood_intent if report.understood_intent else report.query
    query_display = display_query[:120] + "..." if len(display_query) > 120 else display_query
    elements.append(Paragraph(f"<i>Query: {query_display}</i>", body_style))
    elements.append(Spacer(1, 18))
    
    # ===== 2. METRICS SECTION (Synchronized) =====
    elements.append(Paragraph("Executive Summary", section_heading))
    
    # Build metrics table with proper values from state
    metrics_data = [
        ["Metric", "Value", "Status"],
        ["Confidence Score", f"{report.confidence_score:.1%}", "✓"],
        ["Data Quality", f"{report.data_quality_score:.1%}", "✓"],
        ["Sources Analyzed", str(len(report.sources)), "✓"],
        ["Research Iterations", str(report.iteration_count), "✓"],
        ["Hallucination Risk", "Flagged" if report.hallucination_flagged else "Clear", "✓" if not report.hallucination_flagged else "⚠"],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 0.75*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 14))
    
    # ===== 3. SUMMARY =====
    if report.summary:
        elements.append(Paragraph("Summary", section_heading))
        elements.append(Paragraph(report.summary, body_style))
        elements.append(Spacer(1, 12))
    
    # ===== 4. KEY FINDINGS =====
    if report.key_findings:
        elements.append(Paragraph("Key Findings", section_heading))
        for finding in report.key_findings:
            elements.append(Paragraph(f"• {finding}", finding_style))
        elements.append(Spacer(1, 12))
    
    # ===== 5. STRUCTURED TABLES =====
    if report.tables:
        for table_data in report.tables:
            if len(elements) > 10:  # Add page break if content is long
                elements.append(PageBreak())
            
            if table_data.get('title'):
                elements.append(Paragraph(table_data['title'], subsection_heading))
            
            headers = table_data.get('headers', [])
            rows = table_data.get('rows', [])
            
            if headers and rows:
                # Wrap cell content in Paragraphs for word wrap
                table_content = []
                header_paras = [Paragraph(f"<b>{h}</b>", body_style) for h in headers]
                table_content.append(header_paras)
                
                for row in rows:
                    row_paras = [Paragraph(str(cell), body_style) for cell in row]
                    table_content.append(row_paras)
                
                col_widths = [6.5*inch / len(headers)] * len(headers)
                
                data_table = Table(table_content, colWidths=col_widths)
                data_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ]))
                
                elements.append(data_table)
                elements.append(Spacer(1, 12))
    
    # ===== 6. DETAILED SECTIONS =====
    if report.sections:
        for section in report.sections:
            if len(elements) > 15:  # Page management
                elements.append(PageBreak())
            
            if section.get('title'):
                elements.append(Paragraph(section['title'], subsection_heading))
            
            if section.get('content'):
                content = section['content']
                # Clean markdown formatting
                content = content.replace('**', '').replace('##', '').replace('#', '')
                elements.append(Paragraph(content, body_style))
            
            elements.append(Spacer(1, 10))
    
    # ===== 7. SOURCES =====
    if report.sources:
        elements.append(PageBreak())
        elements.append(Paragraph("Sources Consulted", section_heading))
        
        sources_data = [["#", "Title", "URL"]]
        for idx, source in enumerate(report.sources, 1):
            title = source.get("title", "Untitled")
            if len(title) > 80:
                title = title[:77] + "..."
            
            url = source.get("url", "")
            # Wrap long URLs instead of truncating
            url_para = Paragraph(url, ParagraphStyle('UrlStyle', parent=body_style, fontSize=7, leading=8))
            sources_data.append([str(idx), title, url_para])
        
        sources_table = Table(sources_data, colWidths=[0.4*inch, 2.4*inch, 3.7*inch])
        sources_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(sources_table)
        elements.append(Spacer(1, 12))
    
    # ===== 8. VALIDATION RESULTS =====
    if report.validation_results and report.validation_results.get("results"):
        elements.append(Paragraph("Quality Validation Results", section_heading))
        
        validation_data = [["Validation", "Status"]]
        for result in report.validation_results.get("results", []):
            check_type = result.get("validation_type", "").title()
            status = "✓ PASS" if result.get("passed") else "✗ FAIL"
            validation_data.append([check_type, status])
        
        validation_table = Table(validation_data, colWidths=[4*inch, 2.5*inch])
        validation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        elements.append(validation_table)
        elements.append(Spacer(1, 12))
    
    # ===== 9. FOOTER =====
    elements.append(Spacer(1, 12))
    footer_text = f"Report generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} | Deep Research Agent v2.0"
    elements.append(Paragraph(footer_text, footer_style))
    
    # Build PDF
    doc.build(elements)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def generate_pdf_from_state(state: Any) -> bytes:
    """
    Generate PDF from ResearchState with structured data extraction and content curation.
    
    This is the PROPER way: extract structured data from state, NOT from text blob.
    Includes curation to remove debug artifacts and raw retrieval noise.
    """
    
    # 1. Content Curation: Clean final_answer of debug artifacts
    raw_answer = getattr(state, 'final_answer', '')
    
    # Remove common debug/orchestration tags
    curated_answer = re.sub(r'\[SUMMARY:.*?\]', '', raw_answer, flags=re.DOTALL | re.IGNORECASE)
    curated_answer = re.sub(r'\[RESEARCH:.*?\]', '', curated_answer, flags=re.DOTALL | re.IGNORECASE)
    curated_answer = re.sub(r'\[COGNITION:.*?\]', '', curated_answer, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown code fences if they wrap the whole thing or large blocks
    curated_answer = re.sub(r'```(?:markdown|text)?\n(.*?)\n```', r'\1', curated_answer, flags=re.DOTALL)
    
    curated_answer = curated_answer.strip()

    # Extract tables from curated answer
    tables = []
    if curated_answer:
        extracted_tables = MarkdownTableParser.extract_all_tables(curated_answer)
        tables = extracted_tables
    
    # Extract key findings: look for bullet points in curated answer
    key_findings = []
    if curated_answer:
        # Split into lines and find bullet points
        lines = curated_answer.split('\n')
        for line in lines:
            line = line.strip()
            # Match common bullet point styles
            if re.match(r'^[\s]*[-*•][\s]+', line):
                finding = re.sub(r'^[\s]*[-*•][\s]+', '', line).strip()
                # Remove bold marks if present
                finding = finding.replace('**', '')
                if finding and len(finding) > 15:
                    key_findings.append(finding)
        
        # Limit to top 5 substantive findings
        key_findings = key_findings[:5]
    
    # Extract summary: first substantive paragraph from curated answer
    summary = ""
    if curated_answer:
        paragraphs = [p.strip() for p in curated_answer.split('\n\n') if p.strip()]
        for para in paragraphs:
            # Skip headers and tables
            if para and not para.startswith('#') and not para.startswith('|') and len(para) > 60:
                summary = para[:400] + "..." if len(para) > 400 else para
                break
    
    # Extract and deduplicate sources from state
    sources = []
    if hasattr(state, 'scored_sources') and state.scored_sources:
        from utils.domain_filter import normalize_url
        unique_sources = {}
        for s in state.scored_sources:
            # Use canonical normalization for better deduplication
            norm = normalize_url(s.get("url", ""))
            if norm not in unique_sources:
                unique_sources[norm] = s
            elif s.get("overall_score", 0) > unique_sources[norm].get("overall_score", 0):
                # Keep the higher scoring one if duplicate normalized URL
                unique_sources[norm] = s
        sources = list(unique_sources.values())
        # Re-sort by score
        sources.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    
    # Get validation results
    validation_results = getattr(state, 'validation_results', {})
    
    # Create properly curated structured report
    report = ResearchReportDTO(
        query=getattr(state, 'query', ''),
        understood_intent=getattr(state, 'understood_intent', ''),
        summary=summary,
        sections=[], # Main content is handled via tables and findings for better PDF layout
        tables=tables if tables else [],
        key_findings=key_findings,
        confidence_score=getattr(state, 'confidence_score', 0.0),
        data_quality_score=getattr(state, 'data_quality_score', 0.0),
        sources=sources,
        validation_results=validation_results,
        hallucination_flagged=getattr(state, 'hallucination_flagged', False),
        iteration_count=getattr(state, 'iteration', 0),
    )
    
    return generate_pdf_report(report)
