import os
import argparse
import logging
from typing import Tuple, Dict, Any
import shutil

# Note: heavy external imports (tika, fitz) are imported lazily inside functions

# New imports for OCR
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None  # type: ignore
    pytesseract = None  # type: ignore

# Configure a simple logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def extract_text_tika(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Try to extract text and metadata using Apache Tika.

    Returns a tuple (text, metadata). If Tika fails or returns None
    for content, text will be an empty string and metadata will be an empty dict.
    """
    log.info(f"📄 Четене с Apache Tika: {file_path}")
    try:
        # import tika lazily so module import doesn't fail in test environments
        from tika import parser
    except Exception as e:
        log.warning(f"⚠️ Apache Tika is not available: {e}")
        return "", {}

    try:
        parsed = parser.from_file(file_path)
    except Exception as e:
        log.warning(f"⚠️ Tika parsing failed: {e}")
        return "", {}

    text = parsed.get("content") or ""
    metadata = parsed.get("metadata") or {}
    return text.strip(), metadata


def extract_text_fitz(file_path: str) -> str:
    """Extract text using PyMuPDF (fitz). Returns empty string on error."""
    log.info(f"📄 Четене с PyMuPDF (fitz): {file_path}")
    try:
        # import fitz lazily so tests can run without PyMuPDF installed
        import fitz
    except Exception as e:
        log.warning(f"⚠️ PyMuPDF (fitz) is not installed: {e}")
        return ""

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        log.warning(f"⚠️ PyMuPDF failed to open file: {e}")
        return ""

    full_text = []
    try:
        for page in doc:
            # get_text() is robust; collect into list for performance
            full_text.append(page.get_text())
    except Exception as e:
        log.warning(f"⚠️ Error while reading pages with PyMuPDF: {e}")
        # fall through and return whatever we have

    return "".join(full_text).strip()


def extract_text_ocr(file_path: str, max_pages: int = 10, engine: str = 'tesseract', lang: str = 'eng') -> str:
    """Perform OCR on PDF pages using either pytesseract or easyocr.

    engine: 'tesseract' (default) or 'easyocr'. lang is the language code (e.g., 'eng').
    """
    # If pdf2image isn't available, we can't convert PDF pages to images
    if convert_from_path is None:
        log.warning("⚠️ pdf2image is not installed; cannot perform OCR.")
        return ""

    # Only attempt OCR for PDF files
    if not file_path.lower().endswith(".pdf"):
        log.info("⚠️ OCR skipped: not a PDF file")
        return ""

    images = None
    try:
        images = convert_from_path(file_path)
    except Exception as e:
        log.warning(f"⚠️ pdf2image failed to convert PDF to images: {e}")
        return ""

    texts = []

    if engine == 'tesseract':
        if pytesseract is None:
            log.warning("⚠️ pytesseract is not installed. Skipping tesseract OCR.")
            return ""
        # Ensure pytesseract knows where the binary is (works if binary exists but PATH isn't visible)
        tpath = shutil.which('tesseract')
        if tpath:
            try:
                pytesseract.pytesseract.tesseract_cmd = tpath
            except Exception:
                # older pytesseract versions expose different attributes; ignore failures
                pass

        log.info(f"🔎 OCR (tesseract): running on up to {max_pages} pages from {file_path} with lang={lang}")
        for i, img in enumerate(images):
            if i >= max_pages:
                break
            try:
                page_text = pytesseract.image_to_string(img, lang=lang)
            except Exception as e:
                log.warning(f"⚠️ pytesseract failed on page {i}: {e}")
                page_text = ""
            texts.append(page_text)
        return "\n".join(texts).strip()

    elif engine == 'easyocr':
        # lazy import easyocr
        try:
            import easyocr
        except Exception:
            log.warning("⚠️ easyocr is not installed. Install it via `pip install easyocr` to use this engine.")
            return ""
        log.info(f"🔎 OCR (easyocr): running on up to {max_pages} pages from {file_path} with lang={lang}")
        # easyocr wants languages as list of codes, mapping common 'eng'->'en'
        lang_map = {'eng': 'en'}
        easy_lang = lang_map.get(lang, lang)
        try:
            reader = easyocr.Reader([easy_lang], gpu=False)
        except Exception as e:
            log.warning(f"⚠️ easyocr failed to initialize: {e}")
            return ""
        for i, img in enumerate(images):
            if i >= max_pages:
                break
            try:
                # EasyOCR expects a file path, bytes, or numpy array. pdf2image returns PIL Images,
                # so convert to numpy array here to ensure compatibility.
                try:
                    import numpy as _np
                except Exception:
                    _np = None

                img_for_easy = img
                if _np is not None:
                    try:
                        # PIL Image -> numpy array
                        img_for_easy = _np.array(img)
                    except Exception:
                        img_for_easy = img

                result = reader.readtext(img_for_easy)
                page_text = "\n".join([seg[1] for seg in result])
            except Exception as e:
                log.warning(f"⚠️ easyocr failed on page {i}: {e}")
                page_text = ""
            texts.append(page_text)
        return "\n".join(texts).strip()

    else:
        log.warning(f"⚠️ Unknown OCR engine: {engine}. Supported: 'tesseract', 'easyocr'.")
        return ""


def extract_text_hybrid(file_path: str, min_length: int = 50000, ocr: bool = False, ocr_max_pages: int = 10, ocr_engine: str = 'tesseract', ocr_lang: str = 'eng') -> Tuple[str, Dict[str, Any]]:
    """Try Tika first, fall back to PyMuPDF when extracted text is short, and optionally OCR.

    Accepts ocr_engine and ocr_lang to control OCR behavior.
    """
    tika_text, metadata = extract_text_tika(file_path)

    # If Tika returned plenty of content and OCR is not forced, return it
    if len(tika_text) >= min_length and not ocr:
        return tika_text, metadata

    # Otherwise try PyMuPDF
    fitz_text = extract_text_fitz(file_path)
    candidate = tika_text if len(tika_text) >= len(fitz_text) else fitz_text

    # If OCR is requested explicitly, run it and prefer OCR output if present
    if ocr:
        ocr_text = extract_text_ocr(file_path, max_pages=ocr_max_pages, engine=ocr_engine, lang=ocr_lang)
        if ocr_text:
            return ocr_text, metadata
        return candidate, metadata

    # If neither Tika nor fitz produced enough text, try OCR as a fallback
    if len(candidate) < min_length:
        log.info("⚠️ Извлеченият текст е по-кратък от прага — опитваме OCR...")
        ocr_text = extract_text_ocr(file_path, max_pages=ocr_max_pages, engine=ocr_engine, lang=ocr_lang)
        if len(ocr_text) > len(candidate):
            return ocr_text, metadata

    return candidate, metadata


def preview_text(text: str, length: int) -> str:
    """Return a safe preview of text up to `length` characters.

    If the text is shorter than `length`, returns the whole text.
    """
    if not text:
        return ""  # caller will handle messaging
    return text[:length]


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser(description="Extract text from PDF using Tika with a PyMuPDF fallback and show a preview.")
    parser_arg.add_argument("file", help="Path to the PDF (or other) file to extract")
    parser_arg.add_argument("--preview", type=int, default=50000, help="Number of characters to show in the preview (default: 50000)")
    parser_arg.add_argument("--min-length", type=int, default=50000, help="Minimum characters from Tika before skipping PyMuPDF fallback (default: 50000)")
    parser_arg.add_argument("--verbose", action="store_true", help="Show metadata and extra logs")
    parser_arg.add_argument("--ocr", action="store_true", help="Force OCR fallback using pytesseract")
    parser_arg.add_argument("--ocr-max-pages", type=int, default=10, help="Max pages for OCR (default 10)")
    parser_arg.add_argument("--ocr-engine", choices=["tesseract","easyocr"], default="tesseract", help="Which OCR engine to use (tesseract or easyocr)")
    parser_arg.add_argument("--ocr-lang", default="eng", help="Language code for OCR (tesseract lang codes like 'eng', or easyocr codes)")
    args = parser_arg.parse_args()

    if not os.path.exists(args.file):
        log.error("❌ Файлът не съществува: %s", args.file)
        raise SystemExit(1)

    text, meta = extract_text_hybrid(args.file, min_length=args.min_length, ocr=args.ocr, ocr_max_pages=args.ocr_max_pages, ocr_engine=args.ocr_engine, ocr_lang=args.ocr_lang)

    if not text:
        log.warning("❌ Не беше извлечен текст от файла. Може да е сканирано изображение или да има друг проблем.")
        # Provide actionable hint
        log.info("Подсказка: опитайте OCR (pytesseract + pdf2image) ако документът е сканиран.")
        # still print metadata if present and verbose
        if args.verbose and meta:
            log.info("\n📑 Метаданни (Tika):")
            for k, v in meta.items():
                log.info(f"{k}: {v}")
        raise SystemExit(2)

    preview_len = max(0, args.preview)
    preview = preview_text(text, preview_len)

    log.info(f"\n📝 Извлечен текст (първи {preview_len} знака):\n")
    # Print the preview block; keep prints separated from logger for clean raw text
    print(preview)

    if args.verbose:
        log.info("\n📑 Метаданни (Tika):")
        if meta:
            for k, v in meta.items():
                log.info(f"{k}: {v}")
        else:
            log.info("(няма метаданни)")
