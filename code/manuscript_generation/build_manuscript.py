"""Build the SSRN paper in PDF and DOCX from one Markdown and evidence source."""

import argparse
from html import escape
import json
import os
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

sys.dont_write_bytecode = True
from tables import make_tables


SOURCE = Path(__file__).resolve().parent
FONT = "Times New Roman"
PAGE_WIDTH, PAGE_HEIGHT = letter
SIDE_MARGIN = 60
TOP_MARGIN = 52
BOTTOM_MARGIN = 52
CONTENT_WIDTH = PAGE_WIDTH - 2 * SIDE_MARGIN
BODY_SIZE = 11.3
BODY_LEADING = 14.0
CITE = re.compile(r"\[@([A-Za-z0-9]+)(:inline)?\]")
FORMULA = [
    ("R", "it"), (" = ", ""), ("α", "i"), (" + ", ""), ("β", "T,i"), (" T", "it"),
    (" + ", ""), ("β", "P,i"), (" P", "it"), (" + ", ""), ("γ", "i"), (" M", "it"),
    (" + ", ""), ("δ", "i"), (" C", "it"), (" + ", ""), ("ε", "it"),
]


def result_paths(evidence):
    current = (evidence / "exploratory_analysis_results.json",
               evidence / "robustness_and_diagnostics_results.json")
    if all(path.is_file() for path in current):
        return current
    # Support the earlier private archive without renaming it or mixing evidence sets.
    archived = (evidence / "Mission_2_Results.json", evidence / "mission3_results.json")
    if all(path.is_file() for path in archived):
        return archived
    raise FileNotFoundError(f"Provide a complete exploratory and robustness result pair in {evidence}.")


def load_content(source=SOURCE):
    bibliography = json.loads((source / "bibliography.json").read_text(encoding="utf-8"))
    index = {entry["key"]: entry for entry in bibliography}
    manuscript = (source / "manuscript.md").read_text(encoding="utf-8")
    used = set()

    def cite(match):
        key = match[1]
        used.add(key)
        rendered = index[key]["cite"]
        if match[2]:
            rendered = rendered.rsplit(", ", 1)[1]
        return f"({rendered})"

    manuscript = CITE.sub(cite, manuscript)
    if used != set(index):
        raise ValueError(f"Reference mismatch: {sorted(set(index) ^ used)}")
    exploratory_path, robustness_path = result_paths(source / "Evidence")
    exploratory = json.loads(exploratory_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if exploratory["input_sha256"] != robustness["inputs_sha256"]["Master_Dataset_2018_2025.xlsx"]:
        raise ValueError("Exploratory and robustness results do not identify the same master workbook.")
    tables = make_tables(exploratory, robustness)
    blocks = []
    for text in manuscript.strip().split("\n\n"):
        text = " ".join(text.splitlines()).strip()
        if not text:
            continue
        if text == "@references":
            blocks.extend(("reference", ref) for ref in sorted(bibliography, key=lambda item: item["entry"]))
        elif text.startswith("@table "):
            blocks.append(("table", tables[text.split()[1]]))
        elif text in {"@equation", "@page"}:
            blocks.append((text[1:], None))
        elif text.startswith(("@meta ", "@small ")):
            kind, content = text[1:].split(" ", 1)
            blocks.append((kind, content))
        elif text.startswith("### "):
            blocks.append(("subheading", text[4:]))
        elif text.startswith("## "):
            blocks.append(("heading", text[3:]))
        elif text.startswith("# "):
            blocks.append(("title", text[2:]))
        elif text.startswith("@"):
            raise ValueError(f"Unknown manuscript directive: {text}")
        else:
            blocks.append(("body", text))
    return blocks


def pdf_fonts():
    """Use Times New Roman on Windows or compatible Liberation Serif elsewhere."""
    choices = []
    if os.getenv("MANUSCRIPT_FONT_DIR"):
        choices.append(Path(os.environ["MANUSCRIPT_FONT_DIR"]))
    choices.extend((Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts",
                    Path("/usr/share/fonts/truetype/liberation2"),
                    Path("/usr/share/fonts/truetype/liberation")))
    names = (("times.ttf", "timesbd.ttf", "timesi.ttf"),
             ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"))
    for directory in choices:
        for files in names:
            if all((directory / name).is_file() for name in files):
                for name, file in zip(("Paper", "PaperBold", "PaperItalic"), files):
                    pdfmetrics.registerFont(TTFont(name, str(directory / file)))
                pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="PaperBold", italic="PaperItalic")
                return "Paper", "PaperBold", "PaperItalic"
    return "Times-Roman", "Times-Bold", "Times-Italic"


def manuscript_author(blocks):
    return next(text.split(" | ", 1)[0] for kind, text in blocks if kind == "meta")


def build_pdf(blocks, path):
    font, bold, italic = pdf_fonts()
    muted = colors.HexColor("#414141")
    line = colors.HexColor("#969696")
    styles = {
        "body": ParagraphStyle("Body", fontName=font, fontSize=BODY_SIZE, leading=BODY_LEADING, spaceAfter=6,
                               allowWidows=0, allowOrphans=0),
        "title": ParagraphStyle("Title", fontName=bold, fontSize=23, leading=25, spaceAfter=11),
        "meta": ParagraphStyle("Metadata", fontName=font, fontSize=10.5, leading=13, spaceAfter=4),
        "heading": ParagraphStyle("Heading", fontName=bold, fontSize=13.4, leading=16.0,
                                  spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "subheading": ParagraphStyle("Subheading", fontName=bold, fontSize=11.5, leading=14.0,
                                     spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "small": ParagraphStyle("Small", fontName=font, fontSize=9.5, leading=11.5, spaceAfter=6, textColor=muted),
        "reference": ParagraphStyle("Reference", fontName=font, fontSize=10.3, leading=12.4,
                                    leftIndent=14, firstLineIndent=-14, spaceAfter=5),
        "equation": ParagraphStyle("Equation", fontName=italic, fontSize=12.0, leading=17,
                                   alignment=TA_CENTER, spaceBefore=1, spaceAfter=7, keepWithNext=True),
        "caption": ParagraphStyle("Caption", fontName=bold, fontSize=10.6, leading=13.5, spaceBefore=2,
                                  spaceAfter=4, keepWithNext=True),
        "note": ParagraphStyle("Note", fontName=font, fontSize=9.2, leading=11, spaceBefore=4, spaceAfter=8, textColor=muted),
        "cell": ParagraphStyle("Cell", fontName=font, fontSize=10.1, leading=12.2),
        "number": ParagraphStyle("Number", fontName=font, fontSize=10.1, leading=12.2, alignment=TA_RIGHT),
    }
    content = []
    for kind, text in blocks:
        if kind == "reference":
            url = escape(text["url"], quote=True)
            entry = escape(text["entry"])
            display = escape(reference_label(text["url"]))
            content.append(Paragraph(f'{entry} <link href="{url}" color="#284A61">{display}</link>', styles[kind]))
        elif kind == "equation":
            markup = "".join(f"{escape(symbol)}<sub>{escape(subscript)}</sub>" if subscript else escape(symbol)
                             for symbol, subscript in FORMULA)
            content.append(Paragraph(markup, styles[kind]))
        elif kind == "page":
            content.append(PageBreak())
        elif kind == "table":
            rows = []
            for row_index, record in enumerate([text["headers"], *text["rows"]]):
                row = []
                for col, cell in enumerate(record):
                    value = escape(cell)
                    if row_index == 0:
                        value = f"<b>{value}</b>"
                    row.append(Paragraph(value, styles["cell" if col == 0 else "number"]))
                rows.append(row)
            table = Table(rows, colWidths=[CONTENT_WIDTH * w for w in text["widths"]],
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, line),
                ("LINEBELOW", (0, -1), (-1, -1), 0.55, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            content.append(KeepTogether([Paragraph(escape(text["title"]), styles["caption"]), table,
                                         Paragraph(escape(text["note"]), styles["note"])]))
        else:
            content.append(Paragraph(escape(text), styles[kind]))

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont(font, 9.5)
        canvas.drawCentredString(PAGE_WIDTH / 2, 30, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=letter, topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                                 leftMargin=SIDE_MARGIN - 6, rightMargin=SIDE_MARGIN - 6,
                                 title=blocks[0][1], author=manuscript_author(blocks), subject="Exploratory empirical finance working paper")
    document.build(content, onFirstPage=page_number, onLaterPages=page_number)


def reference_label(url):
    """Keep long source links legible while preserving their exact hyperlink target."""
    if url.startswith("https://doi.org/"):
        return "doi:" + url.removeprefix("https://doi.org/")
    labels = {
        "https://open-meteo.com/en/docs/historical-weather-api": "Open-Meteo archive documentation",
        "https://www.mmc.mn/operations/ukhaa-khudag-mine/en": "MMC operations page",
        "https://steppegold.com/17105/": "Steppe Gold release",
    }
    return labels.get(url, "Source workbook and publication record")


def link(paragraph, url, text):
    target = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), target)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "284A61")
    props.append(color)
    run.append(props)
    content = OxmlElement("w:t")
    content.text = text
    run.append(content)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def docx_table(document, block):
    document.add_paragraph(block["title"], "Paper Caption")
    table = document.add_table(rows=0, cols=len(block["headers"]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Pt(CONTENT_WIDTH * width) for width in block["widths"]]
    for column, width in zip(table.columns, widths):
        column.width = width
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "bottom"):
        line = OxmlElement(f"w:{side}")
        line.set(qn("w:val"), "single")
        line.set(qn("w:sz"), "5")
        line.set(qn("w:color"), "444444")
        borders.append(line)
    table._tbl.tblPr.append(borders)
    for row_index, values in enumerate([block["headers"], *block["rows"]]):
        row = table.add_row()
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            properties.append(OxmlElement("w:tblHeader"))
        for index, (cell, width, value) in enumerate(zip(row.cells, widths, values)):
            cell.width = width
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.style = "Paper Cell"
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            run = paragraph.add_run(value)
            run.bold = row_index == 0
    document.add_paragraph(block["note"], "Paper Note")


def build_docx(blocks, path):
    document = Document()
    section = document.sections[0]
    section.page_width = Pt(PAGE_WIDTH)
    section.page_height = Pt(PAGE_HEIGHT)
    section.left_margin = section.right_margin = Pt(SIDE_MARGIN)
    section.top_margin = Pt(TOP_MARGIN)
    section.bottom_margin = Pt(BOTTOM_MARGIN)
    section.footer_distance = Pt(25)
    styles = document.styles

    def format_style(name, size, leading, after=0, before=0, bold=False, keep=False, muted=False):
        if name not in styles:
            styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style = styles[name]
        style.font.name = FONT
        theme_fonts = style.element.get_or_add_rPr().rFonts
        for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            theme_fonts.attrib.pop(qn(f"w:{attribute}"), None)
        paragraph_properties = style.element.get_or_add_pPr()
        border = paragraph_properties.find(qn("w:pBdr"))
        if border is not None:
            paragraph_properties.remove(border)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string("414141" if muted else "000000")
        style.font.bold = bold
        style.font.italic = False
        pf = style.paragraph_format
        pf.line_spacing = Pt(leading)
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.keep_with_next = keep
        pf.widow_control = True
        return style

    format_style("Normal", BODY_SIZE, BODY_LEADING, after=6)
    format_style("Title", 23, 25, after=11, bold=True, keep=True)
    format_style("Heading 1", 13.4, 16, after=5, before=9, bold=True, keep=True)
    format_style("Heading 2", 11.5, 14, after=4, before=6, bold=True, keep=True)
    format_style("Paper Meta", 10.5, 13, after=4)
    format_style("Paper Small", 9.5, 11.5, after=6, muted=True)
    format_style("Paper Reference", 10.3, 12.4, after=5)
    format_style("Paper Equation", 12, 17, after=7, before=1, keep=True)
    format_style("Paper Caption", 10.6, 13.5, after=4, before=2, bold=True, keep=True)
    format_style("Paper Cell", 10.1, 12.2, after=3, before=3, keep=True)
    format_style("Paper Note", 9.2, 11, after=8, before=4, muted=True)
    format_style("Footer", 9.5, 11)
    footer = section.footer.paragraphs[0]
    footer.style = "Footer"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)
    style_names = {"title": "Title", "meta": "Paper Meta", "heading": "Heading 1",
                   "subheading": "Heading 2", "small": "Paper Small", "body": "Normal"}
    for kind, text in blocks:
        if kind == "table":
            docx_table(document, text)
        elif kind == "equation":
            p = document.add_paragraph(style="Paper Equation")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for symbol, subscript in FORMULA:
                p.add_run(symbol).italic = True
                if subscript:
                    run = p.add_run(subscript)
                    run.font.subscript = True
                    run.italic = True
        elif kind == "reference":
            p = document.add_paragraph(text["entry"] + " ", style="Paper Reference")
            p.paragraph_format.left_indent = Pt(14)
            p.paragraph_format.first_line_indent = Pt(-14)
            link(p, text["url"], reference_label(text["url"]))
        elif kind == "page":
            document.add_page_break()
        else:
            document.add_paragraph(text, style=style_names[kind])
    document.core_properties.title = blocks[0][1]
    document.core_properties.author = manuscript_author(blocks)
    document.core_properties.subject = "Exploratory empirical finance working paper"
    document.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local manuscript source and Evidence directory.")
    parser.add_argument("--output-dir", type=Path, default=SOURCE.parents[1] / "replication" / "manuscript")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks = load_content(args.source_dir)
    build_pdf(blocks, args.output_dir / "SSRN_Manuscript.pdf")
    build_docx(blocks, args.output_dir / "SSRN_Manuscript.docx")


if __name__ == "__main__":
    main()
