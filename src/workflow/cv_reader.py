"""
cv_reader.py — Step 1 of CV pipeline: PDF → raw text via vision LLM.

Strategy:
  1. PyMuPDF renders each PDF page to a high-res PIL image (300 DPI).
  2. All page images are sent to Gemini 2.0 Flash (vision-capable) via OpenRouter
     in a single call with a focused extraction prompt.
  3. Returns raw text string — no parsing, no structuring. That is cv_profiler's job.

Notes:
  - Works for any layout: single-column, two-column, scanned, designed CVs.
  - DOCX not supported — must convert to PDF first (e.g. LibreOffice).
  - Uses diskcache to avoid re-calling the LLM for the same PDF file.
  - Retries up to 3 times on transient API errors.
  - Uses OpenRouter (OPENROUTER_API_KEY) — Google AI Studio free tier is disabled on this key.
  - Model: google/gemini-2.0-flash-001 (vision-capable, OpenRouter model ID).
"""

import base64
import hashlib
import logging
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests
from diskcache import Cache
from dotenv import load_dotenv
import os
from PIL import Image
import io

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "google/gemini-2.0-flash-001"  # vision-capable via OpenRouter
DPI = 300                                # render resolution — high enough for dense layouts
MAX_RETRIES = 3
RETRY_DELAY = 2.0                        # seconds between retries
PROMPT_VERSION = "v2"                    # bump when EXTRACTION_PROMPT changes to invalidate cache

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "cv_reader"
_cache = Cache(str(CACHE_DIR))

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_extraction_prompt() -> str:
    return (PROMPTS_DIR / "cv_reader.md").read_text()


# ── Core functions ───────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """Render each page of a PDF to a PIL Image at DPI resolution."""
    images = []
    mat = fitz.Matrix(DPI / 72, DPI / 72)  # 72 dpi is PDF default
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if img.width > 2000 or img.height > 2000:
                img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            images.append(img)
    return images


def _image_to_openrouter_content(img: Image.Image) -> dict:
    """Convert PIL Image to an OpenRouter image_url content block (base64 data URI)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)  # JPEG ~5-10x smaller than PNG; negligible OCR quality loss
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _file_hash(pdf_path: Path) -> str:
    """MD5 hash of the PDF file — used as cache key."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _call_vision_llm(images: list[Image.Image]) -> str:
    """Send all page images + extraction prompt to Gemini via OpenRouter. Returns raw text."""
    extraction_prompt = _load_extraction_prompt()
    content_parts = [{"type": "text", "text": extraction_prompt}]
    content_parts += [_image_to_openrouter_content(img) for img in images]

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content_parts}],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()

            # Defensive JSON parsing — OpenRouter may return error objects on 200
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                error = data.get("error", {})
                msg = error.get("message", str(data)) if isinstance(error, dict) else str(error)
                raise ValueError(f"OpenRouter returned no choices: {msg}")
            text = (choices[0].get("message") or {}).get("content") or ""
            text = text.strip()

            if len(text) < 200:
                # Not retryable — if the model consistently returns short text
                # (blank PDF, refusal), retrying wastes API calls.
                raise ValueError(
                    f"Extracted text too short ({len(text)} chars) — likely failed OCR or model refusal"
                )
            return text

        except ValueError:
            # ValueError = our own logic checks (short text, empty choices) — don't retry
            raise
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"CV extraction failed after {MAX_RETRIES} attempts: {e}") from e
    raise RuntimeError("CV extraction failed — no attempts made")


# ── Public API ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path, use_cache: bool = True) -> str:
    """
    Extract all text from a CV PDF using Gemini 2.0 Flash vision (via OpenRouter).

    Args:
        pdf_path: Path to the PDF file.
        use_cache: If True, returns cached result for same PDF file (by content hash).

    Returns:
        Raw extracted text string.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If the LLM call fails after all retries.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty: {pdf_path.name}")
    try:
        with fitz.open(str(pdf_path)) as _doc:
            if _doc.is_encrypted:
                raise ValueError(f"PDF is password-protected: {pdf_path.name}")
            if len(_doc) == 0:
                raise ValueError(f"PDF has no pages: {pdf_path.name}")
    except fitz.FileDataError as e:
        raise ValueError(f"PDF is corrupted or invalid: {e}") from e

    cache_key = f"cv_text_{MODEL_NAME}_{DPI}_{PROMPT_VERSION}_{_file_hash(pdf_path)}"

    if use_cache and cache_key in _cache:
        logger.info(f"Cache hit for {pdf_path.name}")
        return str(_cache[cache_key])

    logger.info(f"Extracting text from {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
    images = _pdf_to_images(pdf_path)
    logger.info(f"  Rendered {len(images)} page(s) at {DPI} DPI")
    if len(images) > 5:
        logger.warning(f"  Large CV: {len(images)} pages — payload may be substantial")

    raw_text = _call_vision_llm(images)
    logger.info(f"  Extracted {len(raw_text)} chars")

    if use_cache:
        _cache[cache_key] = raw_text

    return raw_text
