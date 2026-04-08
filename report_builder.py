from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(data: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    result = data.get("result", {})

    elements = [
        Paragraph("Automated Plagiarism Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Generated At: {datetime.utcnow().isoformat()} UTC", styles["Normal"]),
        Spacer(1, 12),
    ]

    rows = [
        ["Metric", "Value"],
        ["Classification", str(result.get("classification", "-"))],
        ["Composite Score", f"{float(result.get('composite_score', 0)):.4f}"],
        ["Cosine Similarity", f"{float(result.get('cosine_similarity', 0)):.4f}"],
        ["N-gram Similarity", f"{float(result.get('ngram_similarity', 0)):.4f}"],
        ["Lexical Overlap", f"{float(result.get('lexical_overlap', 0)):.4f}"],
        ["Semantic Similarity", f"{float(result.get('semantic_similarity', 0)):.4f}"],
        ["Precision", f"{float(result.get('precision', 0)):.4f}"],
        ["Recall", f"{float(result.get('recall', 0)):.4f}"],
        ["F1 Score", f"{float(result.get('f1_score', 0)):.4f}"],
    ]

    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 14))

    highlights = data.get("highlights", [])
    if highlights:
        elements.append(Paragraph("Top Matched Phrases", styles["Heading2"]))
        for item in highlights[:10]:
            elements.append(Paragraph(f"- {item.get('phrase', '')}", styles["Normal"]))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
