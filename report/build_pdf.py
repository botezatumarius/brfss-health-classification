"""Render report/report.md to report/report.pdf (Markdown -> HTML -> headless Chrome or Edge).

Usage: python report/build_pdf.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).parent
PDF = DOCS / "report.pdf"
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
)
STYLE = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d2433; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; margin-top: 22pt; border-bottom: 1px solid #c8ccd4; padding-bottom: 3pt; break-before: page; }
h2:first-of-type, h2[id="10-ethics"], h2#references { break-before: auto; }
h3 { font-size: 12pt; margin-top: 16pt; }
h4 { font-size: 10.5pt; margin-top: 12pt; }
h2, h3, h4 { break-after: avoid; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c8ccd4; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f7; padding: 0 2pt; }
pre { background: #f2f4f7; padding: 6pt 8pt; font-size: 8pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 115mm; display: block; margin: 8pt auto 2pt; border: 1px solid #e3e6eb; }
.caption { font-size: 9pt; color: #555; text-align: center; break-before: avoid; }
.toc ul { list-style: none; padding-left: 12pt; margin: 2pt 0; }
.toc > ul { padding-left: 0; }
.toc a { color: inherit; text-decoration: none; }
"""


def browser():
    """The first installed Chromium-based browser (it prints the PDF)."""
    for candidate in BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    raise SystemExit("Chrome or Edge is required to build the PDF.")


def main():
    """Convert the Markdown report to HTML, add the contents, and print it to PDF."""
    converter = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                                  extension_configs={"toc": {"toc_depth": "2-3"}})
    body = converter.convert((DOCS / "report.md").read_text(encoding="utf-8"))
    body = body.replace("<p><em>Figure", '<p class="caption"><em>Figure')
    # Table of contents after the title block.
    first_section = body.index("<h2")
    body = body[:first_section] + f"<p><strong>Contents</strong></p>{converter.toc}" + body[first_section:]
    html = DOCS / "report.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>{STYLE}</style></head>'
                    f"<body>{body}</body></html>", encoding="utf-8")
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.resolve().as_uri()], check=True, capture_output=True)
    html.unlink()
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
