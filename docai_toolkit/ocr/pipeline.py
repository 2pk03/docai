from pathlib import Path
from typing import Iterable

from .clients import OcrClient, PageResult


def run_ocr_to_markdown(pdf_path: Path, output_dir: Path, client: OcrClient) -> Path:
    """Run OCR on a PDF and save combined Markdown output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{pdf_path.stem}.md"
    counter = 1
    while md_path.exists():
        md_path = output_dir / f"{pdf_path.stem}-{counter}.md"
        counter += 1

    pages: Iterable[PageResult] = client.recognize(pdf_path)
    lines: list[str] = []
    for page in pages:
        lines.append(f"# Page {page.page_number}")
        lines.append("")
        lines.append(page.text.strip())
        lines.append("\n")

    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return md_path
