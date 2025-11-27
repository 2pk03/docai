import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path.home() / ".docai" / "config.json"


@dataclass
class OcrConfig:
    provider: str = "deepseek"  # deepseek | tesseract
    api_key: Optional[str] = None
    model: Optional[str] = None  # provider-specific
    endpoint: Optional[str] = None  # user-defined OCR API endpoint


@dataclass
class EmbeddingConfig:
    backend: str = "sentence-transformers"  # sentence-transformers | huggingface-hub
    model: str = "all-mpnet-base-v2"
    device: str = "auto"
    endpoint: Optional[str] = None  # user-defined embedding API endpoint
    api_key: Optional[str] = None  # for hosted endpoints


@dataclass
class LlmConfig:
    backend: str = "huggingface-hub"  # huggingface-hub | local-gguf | openai-compatible
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    api_key: Optional[str] = None
    max_new_tokens: int = 256
    endpoint: Optional[str] = None  # user-defined generation endpoint


@dataclass
class AppConfig:
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    ocr: OcrConfig = field(default_factory=OcrConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls.load_from_file(CONFIG_PATH) or cls()
        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("DOC_AI_HF_TOKEN")
        )
        if hf_token:
            cfg.llm.api_key = hf_token
            cfg.embeddings.api_key = hf_token

        output_dir_env = os.getenv("DOC_AI_OUTPUT_DIR")
        if output_dir_env:
            cfg.output_dir = Path(output_dir_env)
        return cfg

    @classmethod
    def load_from_file(cls, path: Path | None) -> "AppConfig | None":
        if not path:
            return None
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            output_dir=Path(data.get("output_dir", "./outputs")),
            ocr=OcrConfig(**data.get("ocr", {})),
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            llm=LlmConfig(**data.get("llm", {})),
        )

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
