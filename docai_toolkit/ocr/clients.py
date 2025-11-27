from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docai_toolkit.hf_client import HuggingFaceClient


@dataclass
class PageResult:
    page_number: int
    text: str


class OcrClient(ABC):
    @abstractmethod
    def recognize(self, pdf_path: Path) -> List[PageResult]:
        raise NotImplementedError


class DeepSeekOcrClient(OcrClient):
    """Placeholder for DeepSeek OCR API integration."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or "deepseek-ocr-default"

    def recognize(self, pdf_path: Path) -> List[PageResult]:
        raise NotImplementedError("DeepSeek OCR integration not yet implemented.")


class RemoteOcrClient(OcrClient):
    """Generic OCR via a Hugging Face Inference or custom endpoint."""

    def __init__(self, api_key: str | None, endpoint: str, model: str | None = None) -> None:
        self.client = HuggingFaceClient(api_key, default_endpoint=endpoint)
        self.model = model

    def recognize(self, pdf_path: Path) -> List[PageResult]:
        with open(pdf_path, "rb") as handle:
            pdf_bytes = handle.read()

        payload = pdf_bytes
        response = self.client.post_json(payload, content_type="application/pdf")

        # Accept a few possible response shapes.
        if isinstance(response, str):
            pages = [response]
        elif isinstance(response, list):
            pages = [str(item) for item in response]
        elif isinstance(response, dict):
            if "pages" in response:
                pages = [p.get("text", "") if isinstance(p, dict) else str(p) for p in response["pages"]]
            elif "text" in response:
                pages = [str(response["text"])]
            else:
                pages = [str(response)]
        else:
            pages = [""]

        return [PageResult(page_number=i + 1, text=pages[i]) for i in range(len(pages))]


class TesseractOcrClient(OcrClient):
    """Local fallback using pytesseract + pdf2image."""

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_path  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("pytesseract and pdf2image are required for TesseractOcrClient.") from exc

        self._pytesseract = pytesseract
        self._convert_from_path = convert_from_path

    def recognize(self, pdf_path: Path) -> List[PageResult]:
        images = self._convert_from_path(str(pdf_path))
        results: List[PageResult] = []
        for idx, image in enumerate(images):
            text = self._pytesseract.image_to_string(image)
            results.append(PageResult(page_number=idx + 1, text=text))
        return results
