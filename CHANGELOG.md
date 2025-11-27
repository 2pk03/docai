# Changelog

## 0.1.0
### Added
- New `docai_toolkit` package with OCR (local Tesseract or remote endpoint), embeddings/indexing (local or remote), and chat over FAISS.
- GUI updates: OCR → Markdown flow, chat with docs, settings persistence to `~/.docai/config.json`, background threads for OCR/chat, and HF/custom endpoint support.
- Dockerfile (non-root) and GitHub Actions workflows for Docker build and PyPI trusted publishing.
- `pyproject.toml` packaging with console entrypoint `docai-viewer`.
- Tests for PDF writer, remote OCR parsing, and remote embeddings.

### Changed
- Renamed from `docai` to `docai_toolkit` to avoid PyPI naming conflicts.
- Hardened HF client (timeouts/errors), safer FAISS load (no dangerous deserialization), and remote embedding batching.
- README refresh with install, HF onboarding, Docker/XQuartz instructions, and macOS GUI notes.

### Removed
- Legacy training scripts and legacy `pdf_ui.py` entrypoint.
