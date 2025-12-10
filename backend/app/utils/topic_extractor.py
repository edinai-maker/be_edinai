from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from groq import Groq

try:
    import pytesseract
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]

try:
    from pdf2image import convert_from_path
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore[assignment]

try:
    from PyPDF2 import PdfReader
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "PyPDF2 is required for topic extraction. Install it with 'pip install PyPDF2'."
    ) from exc

try:  # pragma: no cover - optional dependency
    from PIL import ImageFile
except ModuleNotFoundError:
    ImageFile = None  # type: ignore[assignment]
else:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import ocrmypdf
    from ocrmypdf.exceptions import MissingDependencyError as OCRMissingDependencyError
    OCRProcessingError = getattr(ocrmypdf.exceptions, "OCRmyPDFError", RuntimeError)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ocrmypdf = None  # type: ignore[assignment]
    OCRMissingDependencyError = RuntimeError  # type: ignore[assignment]
    OCRProcessingError = RuntimeError  # type: ignore[assignment]

try:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "langdetect is required for topic extraction. Install it with 'pip install langdetect'."
    ) from exc

DetectorFactory.seed = 0

logger = logging.getLogger("uvicorn.error").getChild("topic_extractor")

DEFAULT_OCR_LANGUAGE = os.getenv("DEFAULT_OCR_LANGUAGE", "")
GROQ_MODEL = os.getenv("GROQ_TOPIC_MODEL", "openai/gpt-oss-120b")
MAX_INPUT_CHARS = 15_000
TOPIC_EXTRACTION_PROMPT_TEMPLATE = (
    "Read the PDF and answer in  {language_label}. "
    "do not give me answer outof pdf and topic and subtopic."
    "give it chapter wise."
    "Identify the main textbook topics as numbered items (1., 2., 3., ...). "
    "If a topic has subtopics, list them as bullet points under that topic. "
    "Do not create a generic topic such as 'Topics' or 'All Topics'; instead, make every real chapter/topic a main numbered item. "
    "For each subtopic, include a clear narration using the original textbook sentences as much as possible, "
    "but remove page numbers or page references (for example: 'Page 12', 'p. 12'). Avoid copying header/footer text. "
    "Do not wrap titles in Markdown markers such as ** or ##. "
    "Use this exact layout:\n"
    "1. Topic Title\n"
    "- Subtopic Title: narration text covering all relevant sentences\n"
    "- Another Subtopic: narration text (continue as needed).\n"
    "If a topic has no subtopics, place the narration directly on the line after the numbered topic."
)

LANGUAGE_SPECS: Dict[str, Dict[str, Any]] = {
    "guj": {
        "label": "ગુજરાતી",
        "script_pattern": re.compile(r"[\u0A80-\u0AFF]"),
        "ocr_code": "guj",
        "min_ratio": 0.2,
    },
    "hin": {
        "label": "हिंदी",
        "script_pattern": re.compile(r"[\u0900-\u097F]"),
        "ocr_code": "hin",
        "min_ratio": 0.2,
    },
    "eng": {
        "label": "English",
        "script_pattern": re.compile(r"[A-Za-z]"),
        "ocr_code": "eng",
        "min_ratio": 0.3,
    },
}

LANGDETECT_TO_LANGUAGE_CODE = {
    "hi": "hin",
    "gu": "guj",
    "en": "eng",
}

HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\s+(?P<title>.+)$")

PAGE_REF_PATTERN = re.compile(
    r"\b(?:page|pg\.?|p\.)\s*\d+\b",
    re.IGNORECASE,
)

GENERIC_TOPIC_TITLES = {
    "topics",
    "all topics",
    "topic list",
    "list of topics",
}


def _select_supported_language(code: Optional[str]) -> Optional[str]:
    if not code:
        return None

    candidates: List[str] = []
    candidates.extend(re.split(r"[+,\s]+", code))
    candidates.append(code)

    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized in LANGUAGE_SPECS:
            return normalized

    return None


def _get_language_spec(code: Optional[str]) -> Dict[str, Any]:
    selected = _select_supported_language(code)
    if not selected:
        raise ValueError(f"Unsupported or missing language code: {code!r}")
    return LANGUAGE_SPECS[selected]


def _count_alpha_chars(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _script_ratio(text: str, *, pattern: re.Pattern[str]) -> float:
    alpha_total = _count_alpha_chars(text)
    if alpha_total == 0:
        return 0.0

    matches = pattern.findall(text)
    return len(matches) / alpha_total if alpha_total else 0.0


def _guess_language_by_script(text: str) -> Optional[str]:
    best_code: Optional[str] = None
    best_ratio = 0.0

    for code, spec in LANGUAGE_SPECS.items():
        ratio = _script_ratio(text, pattern=spec["script_pattern"])
        if ratio >= float(spec["min_ratio"]) and ratio > best_ratio:
            best_code = code
            best_ratio = ratio

    return best_code


def read_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text_parts: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        text_parts.append(f"\n--- Page {page_number} ---\n{page_text.strip()}\n")

    return "\n".join(text_parts)


def _filter_text_by_language(text: str, *, spec: Dict[str, Any]) -> str:
    pattern = spec["script_pattern"]
    min_ratio = float(spec["min_ratio"])
    filtered_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        ratio = _script_ratio(line, pattern=pattern)
        if ratio >= min_ratio:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _prepare_excerpt(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _prepare_model_input(text: str, *, spec: Dict[str, Any]) -> str:
    filtered = _filter_text_by_language(text, spec=spec)
    candidate = filtered or text
    return _prepare_excerpt(candidate, limit=MAX_INPUT_CHARS)


def _build_topic_prompt(language_label: str) -> str:
    return TOPIC_EXTRACTION_PROMPT_TEMPLATE.format(language_label=language_label)


def extract_numbered_headings(text: str, *, max_items: int = 50) -> List[Tuple[str, str]]:
    headings: List[Tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 4:
            continue

        match = HEADING_PATTERN.match(line)
        if not match:
            continue

        number = match.group("number")
        title = match.group("title").strip(" -.:")
        if not title:
            continue

        headings.append((number, title))
        if len(headings) >= max_items:
            break

    return headings


def _merge_unique_titles(*title_groups: Iterable[str]) -> List[str]:
    """Merge multiple title lists while preserving order and removing duplicates."""

    seen: set[str] = set()
    merged: List[str] = []

    for titles in title_groups:
        for raw_title in titles:
            if not raw_title:
                continue
            normalized = re.sub(r"\s+", " ", raw_title).strip(" -–—:•\t")
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)

    return merged


def stream_topics_from_text(
    pdf_text: str,
    client: Groq,
    *,
    language_code: Optional[str] = None,
) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Groq client is not configured.")

    spec = _get_language_spec(language_code)
    language_label = spec["label"]
    prompt = _build_topic_prompt(language_label)
    model_input = _prepare_model_input(pdf_text, spec=spec)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": model_input},
    ]

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=12000,
        )
    except Exception as exc:  # pragma: no cover - network dependency
        raise RuntimeError(f"GROQ topic extraction failed: {exc}") from exc

    if not completion.choices:
        raise RuntimeError("No response received from GROQ topic extraction.")

    content = (completion.choices[0].message.content or "").strip()
    headings = extract_numbered_headings(content)

    return {
        "content": content,
        "headings": headings,
        "language_label": language_label,
    }


def detect_ocr_language_with_pytesseract(pdf_path: Path) -> Optional[str]:
    if pytesseract is None or convert_from_path is None:
        return None

    try:
        images = convert_from_path(str(pdf_path), first_page=1, last_page=1)
    except Exception:
        return None

    if not images:
        return None

    try:
        osd = pytesseract.image_to_osd(images[0], output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractError, RuntimeError):
        return None

    script = (osd.get("script") or "").strip().lower()
    script_to_language = {
        "devanagari": "hin",
        "gujarati": "guj",
        "latin": "eng",
    }

    language_code = script_to_language.get(script)
    if language_code and language_code in LANGUAGE_SPECS:
        return language_code

    return None


def detect_dominant_language(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        logger.info("No text found; unable to detect language.")
        return None

    script_guess = _guess_language_by_script(stripped)
    if script_guess:
        spec = LANGUAGE_SPECS[script_guess]
        logger.info("Detected language via script analysis: %s (%s)", script_guess, spec["label"])
        return script_guess

    sample = stripped[:MAX_INPUT_CHARS]
    try:
        guesses = detect_langs(sample)
    except LangDetectException:
        logger.info("langdetect could not determine the language for the provided text sample.")
        return None

    for guess in guesses:
        internal_code = LANGDETECT_TO_LANGUAGE_CODE.get(guess.lang)
        if internal_code:
            spec = LANGUAGE_SPECS[internal_code]
            logger.info(
                "Detected language via langdetect (%s, prob=%.2f): %s (%s)",
                guess.lang,
                getattr(guess, "prob", 0.0),
                internal_code,
                spec["label"],
            )
            return internal_code

    logger.info("Unable to detect language from the provided text sample.")
    return None


def read_pdf_with_ocrmypdf(pdf_path: Path, ocr_language: str) -> str:
    if ocrmypdf is None:
        raise RuntimeError("ocrmypdf is required. Install it with 'pip install ocrmypdf'.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_pdf = tmp_path / "ocr_output.pdf"
        sidecar_text = tmp_path / "ocr_output.txt"

        try:
            ocrmypdf.ocr(
                str(pdf_path),
                str(output_pdf),
                sidecar=str(sidecar_text),
                force_ocr=True,
                language=ocr_language,
                progress_bar=False,  # progress bar / extra output band
                rotate_pages=True,
                deskew=True,
                # oversample=300,  # OPTIONAL: DPI normalize karne ke liye, chaho to uncomment karo
            )
        except OCRMissingDependencyError as exc:
            raise RuntimeError(
                "OCRmyPDF is missing required external dependencies (e.g., Ghostscript or Tesseract). "
                "Install them and try again."
            ) from exc
        except getattr(ocrmypdf.exceptions, "PriorOcrFoundError", tuple()) as exc:  # type: ignore[arg-type]
            logger.warning("OCRmyPDF detected existing OCR layer; skipping additional OCR.")
            return ""
        except getattr(ocrmypdf.exceptions, "ExitCodeError", tuple()) as exc:  # type: ignore[arg-type]
            raise RuntimeError(f"OCRmyPDF failed: {exc}") from exc
        except OCRProcessingError as exc:  # type: ignore[misc]
            raise RuntimeError(f"OCRmyPDF error: {exc}") from exc

        if sidecar_text.exists():
            sidecar_content = sidecar_text.read_text(encoding="utf-8", errors="ignore")
            if sidecar_content.strip():
                return sidecar_content

        if output_pdf.exists():
            try:
                return read_pdf(output_pdf)
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.warning("Failed to read OCR output PDF text: %s", exc)

        return ""


def extract_text_with_auto_language(pdf_path: Path) -> Tuple[str, Optional[str]]:
    """
    Prefer embedded PDF text first (fast + clean),
    and only fall back to OCR when needed.
    """

    # 1) Try normal text extraction first (jaise local me ho raha tha)
    try:
        raw_text = read_pdf(pdf_path)
    except Exception as exc:
        logger.warning("Failed to read embedded text from %s: %s", pdf_path.name, exc)
        raw_text = ""

    if raw_text.strip():
        logger.info("Using embedded PDF text for %s (no OCR needed).", pdf_path.name)
        detected_language = detect_dominant_language(raw_text)
        return raw_text, detected_language

    logger.info(
        "Embedded text empty for %s; falling back to OCR with ocrmypdf.",
        pdf_path.name,
    )

    # 2) Agar embedded text nahi mila, tab hi OCR use karo
    def _run_ocr(lang_code: str) -> str:
        try:
            # map internal lang code -> tesseract code
            lang_map = {
                "eng": "eng",
                "hin": "hin",
                "guj": "guj",
            }
            ocr_language = lang_map.get(lang_code, "eng+hin+guj")
            return read_pdf_with_ocrmypdf(pdf_path, ocr_language=ocr_language)
        except RuntimeError as exc:  # ocrmypdf / deps issue
            logger.warning("OCR failed for %s using %s: %s", pdf_path.name, lang_code, exc)
            return ""

    detected_osd_language = detect_ocr_language_with_pytesseract(pdf_path)
    candidate_order: List[str] = []
    if detected_osd_language:
        candidate_order.append(detected_osd_language)

    default_language = _select_supported_language(DEFAULT_OCR_LANGUAGE)
    if default_language and default_language not in candidate_order:
        candidate_order.append(default_language)

    for code in LANGUAGE_SPECS.keys():
        if code not in candidate_order:
            candidate_order.append(code)

    ocr_text = ""
    ocr_language_used: Optional[str] = None
    best_attempt_text = ""
    best_attempt_language: Optional[str] = None
    best_attempt_ratio = 0.0

    for candidate in candidate_order:
        attempt = _run_ocr(candidate)
        if not attempt.strip():
            continue

        spec = LANGUAGE_SPECS.get(candidate)
        ratio = 0.0
        if spec:
            ratio = _script_ratio(attempt, pattern=spec["script_pattern"])
            if ratio < float(spec["min_ratio"]):
                if ratio > best_attempt_ratio:
                    best_attempt_text = attempt
                    best_attempt_language = candidate
                    best_attempt_ratio = ratio
                logger.info(
                    "OCR text for %s using %s model did not meet required script ratio (%.2f < %.2f); trying next candidate.",
                    pdf_path.name,
                    candidate,
                    ratio,
                    float(spec["min_ratio"]),
                )
                continue

        ocr_text = attempt
        ocr_language_used = candidate
        break

    if not ocr_text.strip() and best_attempt_text.strip():
        ocr_text = best_attempt_text
        ocr_language_used = best_attempt_language

    if not ocr_text.strip():
        logger.info("OCR attempts did not produce text for %s; returning empty.", pdf_path.name)
        return "", None

    detected_language = detect_dominant_language(ocr_text)
    logger.info("Final Detected Language (after OCR): %s", detected_language)

    if detected_language and detected_language != ocr_language_used:
        refined = _run_ocr(detected_language)
        if refined.strip():
            ocr_text = refined
            ocr_language_used = detected_language

    final_language = detected_language or ocr_language_used
    return ocr_text, final_language

    
def detect_pdf_language(pdf_path: Path) -> Dict[str, str]:
    """Detect the probable language code/label for the given PDF without OCR."""

    pdf_path = Path(pdf_path)

    try:
        raw_text = read_pdf(pdf_path)
    except Exception:
        raw_text = ""

    language_code = detect_dominant_language(raw_text)
    pytesseract_language = detect_ocr_language_with_pytesseract(pdf_path)
    effective_language = pytesseract_language or language_code

    if not effective_language:
        return {
            "language_code": "",
            "language_label": "Unknown",
        }

    try:
        spec = _get_language_spec(effective_language)
    except ValueError:
        logger.warning(
            "Detected unsupported language '%s' while analyzing %s",
            effective_language,
            pdf_path.name,
        )
        return {
            "language_code": "",
            "language_label": "Unknown",
        }

    return {
        "language_code": effective_language,
        "language_label": spec["label"],
    }


def parse_topics_text(topics_text: str) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []
    current_topic: Optional[Dict[str, Any]] = None
    current_subtopic: Optional[Dict[str, Any]] = None

    topic_pattern = re.compile(r"^(\d+)([\).\-:]|\s)+")

    def _split_subtopic(text: str) -> Tuple[str, str]:
        cleaned = text.strip()
        if not cleaned:
            return "", ""

        separators = [":", " - ", " – ", " — ", " —", " –", "-", "—"]
        for separator in separators:
            if separator in cleaned:
                parts = cleaned.split(separator, 1)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
        return cleaned, ""

    def _clean_title(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\*+|\*+$", "", cleaned)
        return cleaned.strip()

    def _clean_narration(text: str) -> str:
        if not text:
            return ""
        cleaned = PAGE_REF_PATTERN.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    for raw_line in topics_text.splitlines():
        line = raw_line.strip()
        if not line:
            current_subtopic = None
            continue

        number_match = topic_pattern.match(line)
        if number_match:
            title = topic_pattern.sub("", line).strip()
            if not title:
                title = line.strip()

            title = _clean_title(title)

            current_topic = {
                "title": title,
                "summary": "",
                "subtopics": [],
            }
            topics.append(current_topic)
            current_subtopic = None
            continue

        if line[0] in {"-", "•", "*"} and current_topic:
            subtopic_text = line.lstrip("-•* \t").strip()
            subtopic_title, subtopic_narration = _split_subtopic(subtopic_text)
            subtopic_title = _clean_title(subtopic_title)
            subtopic_narration = _clean_narration(subtopic_narration)
            current_subtopic = {
                "title": subtopic_title,
                "narration": subtopic_narration,
            }
            current_topic["subtopics"].append(current_subtopic)
            continue

        if current_subtopic is not None:
            narration = current_subtopic.get("narration", "")
            combined = f"{narration} {line}".strip() if narration else line
            current_subtopic["narration"] = _clean_narration(combined)
            continue

        if current_topic:
            summary = current_topic.get("summary", "")
            combined = f"{summary} {line}".strip() if summary else line
            current_topic["summary"] = combined
        else:
            current_topic = {
                "title": _clean_title(line),
                "summary": "",
                "subtopics": [],
            }
            topics.append(current_topic)
            current_subtopic = None

    for topic in topics:
        title_text = topic.get("title", "")
        topic["title"] = _clean_title(title_text)

        summary_text = topic.get("summary", "").strip()
        summary_text = PAGE_REF_PATTERN.sub("", summary_text)
        topic["summary"] = re.sub(r"\s+", " ", summary_text)

        cleaned_subtopics: List[Dict[str, str]] = []
        seen_keys = set()
        for subtopic in topic.get("subtopics", []):
            if isinstance(subtopic, dict):
                title = _clean_title(subtopic.get("title", ""))
                narration = _clean_narration(subtopic.get("narration", ""))
            else:
                title = _clean_title(str(subtopic))
                narration = ""

            if not title and not narration:
                continue

            key = (title.lower(), narration.lower())
            if key in seen_keys:
                continue

            seen_keys.add(key)
            cleaned_subtopics.append({
                "title": title,
                "narration": narration,
            })

        topic["subtopics"] = cleaned_subtopics

    flattened_topics: List[Dict[str, Any]] = []
    for topic in topics:
        title = topic.get("title", "").strip()
        normalized = title.lower()
        if normalized in GENERIC_TOPIC_TITLES and topic.get("subtopics"):
            for subtopic in topic["subtopics"]:
                sub_title = subtopic.get("title", "").strip()
                sub_narration = subtopic.get("narration", "").strip()
                if not sub_title and not sub_narration:
                    continue
                flattened_topics.append(
                    {
                        "title": sub_title,
                        "summary": sub_narration,
                        "subtopics": [],
                    }
                )
        else:
            flattened_topics.append(topic)

    return flattened_topics


def extract_topics_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Reading PDF for topic extraction: %s", pdf_path.name)
    pdf_text, language_code = extract_text_with_auto_language(pdf_path)

    if not pdf_text.strip():
        logger.error("No text could be extracted from %s; skipping topic extraction.", pdf_path.name)
        return {
            "success": False,
            "language_code": "",
            "language_label": "Unknown",
            "topics_text": "",
            "headings": [],
            "chapter_titles": [],
            "chapter_title": "",
            "excerpt": "",
            "error": "Unable to read any text from the PDF; topics were not generated.",
        }

    # Extract chapter titles from PDF headings before processing
    chapter_headings = extract_numbered_headings(pdf_text)
    chapter_titles_pdf = [title.strip() for number, title in chapter_headings if title.strip()]
    chapter_titles = chapter_titles_pdf.copy()
    primary_chapter_title = chapter_titles[0] if chapter_titles else ""

    # Print chapter titles to terminal (PDF-derived)
    if chapter_titles_pdf:
        print("\n" + "="*50)
        print("CHAPTER TITLES EXTRACTED FROM PDF:")
        print("="*50)
        for i, title in enumerate(chapter_titles_pdf, 1):
            print(f"{i}. {title}")
        print("="*50 + "\n")
        logger.info(
            "Extracted %d chapter titles from PDF: %s",
            len(chapter_titles_pdf),
            ", ".join(chapter_titles_pdf[:3]) + ("..." if len(chapter_titles_pdf) > 3 else ""),
        )
    else:
        print("\n" + "="*50)
        print("NO CHAPTER TITLES FOUND IN PDF HEADINGS")
        print("="*50 + "\n")
        logger.info("No numbered chapter headings found in PDF")

    if not language_code:
        fallback_language = _select_supported_language(DEFAULT_OCR_LANGUAGE) or "eng"
        logger.warning(
            "Language detection failed for %s; falling back to %s for topic extraction.",
            pdf_path.name,
            fallback_language,
        )
        language_code = fallback_language

    language_spec = _get_language_spec(language_code)
    logger.info(
        "Detected PDF language: %s (%s)",
        language_code,
        language_spec["label"],
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    stream_result = stream_topics_from_text(pdf_text, client, language_code=language_code)

    spec = language_spec
    parsed_topics = parse_topics_text(stream_result["content"])
    llm_headings = stream_result.get("headings", [])
    chapter_titles_llm = [title.strip() for number, title in llm_headings if title.strip()]
    merged_chapter_titles = _merge_unique_titles(chapter_titles_llm, chapter_titles)
    chapter_titles = merged_chapter_titles or chapter_titles
    if chapter_titles:
        primary_chapter_title = chapter_titles[0]

    # Print final topics with chapter titles
    print("\n" + "="*50)
    print("TOPIC EXTRACTION COMPLETED:")
    print("="*50)

    print(f"Language: {spec['label']} ({language_code})")
    print(f"Chapter Titles Found: {len(chapter_titles)}")
    print(f"Topics Extracted: {len(parsed_topics)}")
    if chapter_titles:
        print("Chapter Titles:", " | ".join(chapter_titles[:3]) + ("..." if len(chapter_titles) > 3 else ""))
    print("="*50 + "\n")

    return {
        "success": bool(stream_result["content"].strip()),
        "language_code": language_code,
        "language_label": spec["label"],
        "topics_text": stream_result["content"],
        "headings": stream_result["headings"],
        "chapter_titles": chapter_titles,
        "chapter_title": primary_chapter_title,
        "excerpt": _prepare_excerpt(pdf_text, limit=1_000),
        "topics": parsed_topics,
    }