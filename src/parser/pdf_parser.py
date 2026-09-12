"""
pdf_parser.py
=============

Owner: Person 2 (PDF / resume data-processing pipeline)

Deterministic, local-only resume parsing pipeline:

    PDF resumes
        -> extract_text
        -> clean_text
        -> detect_sections
        -> extract_skills
        -> extract_contact_info
        -> structured resume dict

No LLMs, no external APIs, no cloud calls. Everything here uses PyMuPDF
(fitz) and the Python standard library only.

Public API (used by Person 1 / ranking and Person 4 / explainability):

    extract_text(pdf_path) -> str
    clean_text(text) -> str
    detect_sections(text) -> dict
    extract_skills(text) -> list[str]
    extract_contact_info(text) -> dict
    process_resume(pdf_path, resume_id=None) -> dict
    process_all_resumes(resumes_dir=..., output_path=...) -> list[dict]
    process_job_description(path) -> dict
"""

from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - import guard
    fitz = None
    _FITZ_IMPORT_ERROR = exc
else:
    _FITZ_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Paths (relative to project root, computed from this file's location so it
# works regardless of the current working directory).
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_RESUMES_DIR = os.path.join(PROJECT_ROOT, "data", "resumes")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "resumes.json")
DEFAULT_JD_PATH = os.path.join(PROJECT_ROOT, "data", "job_description.txt")


# ---------------------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> dict:
    """
    Extract raw text from every page of a PDF using PyMuPDF.

    Returns a dict rather than a bare string so callers can distinguish
    "no extractable text" and "failed to open" without exceptions leaking
    into the batch pipeline:

        {"text": "...", "status": "success"}
        {"text": "",    "status": "no_text"}
        {"text": "",    "status": "error", "error": "..."}
    """
    if fitz is None:
        return {
            "text": "",
            "status": "error",
            "error": f"PyMuPDF (fitz) is not installed: {_FITZ_IMPORT_ERROR}",
        }

    if not os.path.isfile(pdf_path):
        return {"text": "", "status": "error", "error": f"File not found: {pdf_path}"}

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # malformed / encrypted / unreadable PDF
        return {"text": "", "status": "error", "error": f"Unable to open PDF: {exc}"}

    pages_text = []
    try:
        for page in doc:
            try:
                page_text = page.get_text("text")
            except Exception as exc:
                # One bad page shouldn't kill the whole document.
                page_text = ""
            pages_text.append(page_text or "")
    finally:
        doc.close()

    full_text = "\n".join(pages_text)

    if not full_text.strip():
        return {"text": "", "status": "no_text"}

    return {"text": full_text, "status": "success"}


# ---------------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------------

# Terms with punctuation/casing that naive cleaning could mangle. These are
# protected by construction: our cleaning only touches whitespace and control
# characters, never letters/punctuation within tokens, so C++, C#, .NET,
# Node.js, React.js, Next.js, Vue.js, GitHub, AWS, GCP, REST API and
# "Machine Learning" survive untouched. Kept here as living documentation /
# a regression checklist for tests.
PROTECTED_TERMS_SAMPLE = [
    "C++", "C#", ".NET", "Node.js", "React.js", "Next.js", "Vue.js",
    "GitHub", "AWS", "GCP", "REST API", "Machine Learning",
]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """
    Deterministic text cleaning that preserves technical terms and section
    boundaries. Only whitespace / control characters are normalized -
    nothing that would corrupt tokens like C++, .NET, REST API, etc.
    """
    if not text:
        return ""

    # Normalize unicode (NFC) without stripping accented/non-ASCII chars.
    text = unicodedata.normalize("NFC", text)

    # Remove null/control characters (keep \n, \t for structure - \t gets
    # collapsed below anyway).
    text = _CONTROL_CHARS_RE.sub("", text)

    # Normalize Windows/Mac line endings to \n.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of spaces/tabs (but not newlines) to a single space.
    text = _MULTI_SPACE_RE.sub(" ", text)

    # Strip trailing whitespace at the end of each line.
    text = _TRAILING_SPACE_RE.sub("\n", text)

    # Collapse 3+ blank lines down to a single blank line (preserves section
    # boundaries without leaving huge gaps).
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)

    # Strip each line's leading/trailing spaces individually, then rejoin.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ---------------------------------------------------------------------------
# 3. RESUME SECTION DETECTION
# ---------------------------------------------------------------------------

SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "professional summary", "career objective"],
    "skills": ["skills", "technical skills", "technical expertise", "core competencies", "key skills"],
    "experience": [
        "experience", "work experience", "professional experience",
        "internship", "internships", "employment history", "work history",
    ],
    "education": ["education", "academic background", "academic qualifications", "educational qualifications"],
    "projects": ["projects", "academic projects", "personal projects", "project experience"],
    "certifications": ["certifications", "certificates", "licenses & certifications"],
    "achievements": ["achievements", "accomplishments", "awards", "honors and awards"],
}

# Build a flat alias -> canonical lookup, longest alias first so e.g.
# "technical skills" matches before a bare "skills" heuristic could.
_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in SECTION_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical

_SORTED_ALIASES = sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)

# A line is considered a "heading" if, once stripped of punctuation/colons
# and lowercased, it exactly matches a known alias and is short (headings
# are not full sentences).
_HEADING_STRIP_RE = re.compile(r"[:\-–—•.]+$")


def _line_is_heading(line: str) -> Optional[str]:
    candidate = line.strip()
    if not candidate or len(candidate) > 40:
        return None
    candidate = _HEADING_STRIP_RE.sub("", candidate).strip().lower()
    if candidate in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[candidate]
    return None


def detect_sections(text: str) -> dict:
    """
    Lightweight deterministic section detection based on a heading-alias
    dictionary. Scans line-by-line for lines that are (close to) an exact
    match for a known section heading, and assigns all text between two
    headings to the preceding section.

    Missing sections return "" rather than raising/crashing.
    """
    result = {canonical: "" for canonical in SECTION_ALIASES}

    if not text:
        return result

    lines = text.split("\n")
    current_section = None
    buffers = {canonical: [] for canonical in SECTION_ALIASES}

    for line in lines:
        heading = _line_is_heading(line)
        if heading:
            current_section = heading
            continue
        if current_section is not None:
            buffers[current_section].append(line)

    for canonical, buf in buffers.items():
        result[canonical] = "\n".join(buf).strip()

    return result


# ---------------------------------------------------------------------------
# 4. DETERMINISTIC SKILL EXTRACTION
# ---------------------------------------------------------------------------

# canonical -> list of aliases (canonical itself is matched too).
SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "C": ["c"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c sharp"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Node.js": ["node.js", "nodejs", "node js", "node"],
    "Express": ["express", "express.js", "expressjs"],
    "Next.js": ["next.js", "nextjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring": ["spring", "spring boot", "springboot"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud platform", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Git": ["git"],
    "GitHub": ["github"],
    "REST API": ["rest api", "restful api", "rest"],
    "GraphQL": ["graphql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Linux": ["linux"],
    "Firebase": ["firebase"],
    "Tailwind": ["tailwind", "tailwindcss", "tailwind css"],
    "Bootstrap": ["bootstrap"],
}

# Build alias -> canonical, sorted longest-first so multi-word aliases
# ("google cloud platform") are matched before shorter ones ("gcp") could
# short-circuit, and so overlapping aliases resolve to the more specific
# canonical term.
_SKILL_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    for _alias in _aliases:
        _SKILL_ALIAS_TO_CANONICAL[_alias.lower()] = _canonical

_SORTED_SKILL_ALIASES = sorted(_SKILL_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)


def _compile_skill_pattern(alias: str) -> re.Pattern:
    """
    Build a word-boundary-safe regex for a skill alias. Standard \\b doesn't
    work well for tokens containing '+', '#', '.' (e.g. C++, C#, Node.js)
    because those aren't word characters, so \\b sits in the wrong place.
    Instead we require that the alias not be immediately preceded/followed
    by an alphanumeric character.
    """
    escaped = re.escape(alias)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_SKILL_PATTERNS = {alias: _compile_skill_pattern(alias) for alias in _SORTED_SKILL_ALIASES}


def extract_skills(text: str) -> list:
    """
    Deterministic skill extraction via alias matching with safe boundaries
    (no naive substring matches - e.g. the letter "c" inside "vaccine" will
    not match the "C" language). Returns a sorted, de-duplicated list of
    canonical skill names.
    """
    if not text:
        return []

    found = set()
    for alias in _SORTED_SKILL_ALIASES:
        pattern = _SKILL_PATTERNS[alias]
        if pattern.search(text):
            found.add(_SKILL_ALIAS_TO_CANONICAL[alias])

    return sorted(found)


# ---------------------------------------------------------------------------
# 5. BASIC CONTACT EXTRACTION
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Covers common Indian (+91, 10-digit) and general international formats:
# optional +country code, optional separators (space/dash/dot), 10-13 digits.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}"
)


def _looks_like_valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 10 <= len(digits) <= 13


def _extract_name(text: str) -> Optional[str]:
    """
    Heuristic only: take the first non-empty line before any known section
    heading, provided it looks like a name (short, no digits, no @, not a
    heading itself). Returns None rather than guessing if nothing plausible
    is found.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:8]:  # names live at the very top of a resume
        if _line_is_heading(line):
            break
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        word_count = len(line.split())
        if 1 <= word_count <= 5 and len(line) <= 60:
            return line
    return None


def extract_contact_info(text: str) -> dict:
    """
    Deterministic regex/heuristic contact extraction. Returns None for any
    field that can't be confidently identified - never fabricates values.
    """
    if not text:
        return {"name": None, "email": None, "phone": None}

    email_match = _EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else None

    phone = None
    for candidate in _PHONE_RE.findall(text):
        if _looks_like_valid_phone(candidate):
            phone = candidate.strip()
            break

    name = _extract_name(text)

    return {"name": name, "email": email, "phone": phone}


# ---------------------------------------------------------------------------
# 6/7. STRUCTURED RESUME OBJECT + BATCH PROCESSING
# ---------------------------------------------------------------------------

def _empty_sections() -> dict:
    return {canonical: "" for canonical in SECTION_ALIASES}


def process_resume(pdf_path: str, resume_id: Optional[str] = None) -> dict:
    """
    Run the full pipeline on a single PDF and return a structured resume
    dict. Never raises - failures are represented in parsing_status/error.
    """
    filename = os.path.basename(pdf_path)
    if resume_id is None:
        resume_id = os.path.splitext(filename)[0]

    base_result = {
        "id": resume_id,
        "filename": filename,
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "sections": _empty_sections(),
        "raw_text": "",
        "parsing_status": "error",
        "error": None,
    }

    extraction = extract_text(pdf_path)

    if extraction["status"] == "error":
        base_result["error"] = extraction.get("error", "Unknown extraction error")
        base_result["parsing_status"] = "error"
        return base_result

    if extraction["status"] == "no_text":
        base_result["parsing_status"] = "no_text"
        return base_result

    raw_text = extraction["text"]
    cleaned = clean_text(raw_text)

    try:
        sections = detect_sections(cleaned)
        skills = extract_skills(cleaned)
        contact = extract_contact_info(cleaned)
    except Exception as exc:
        # Defensive: a bug in downstream steps shouldn't lose the raw text
        # or crash the batch - still report what succeeded (extraction).
        base_result["raw_text"] = raw_text
        base_result["parsing_status"] = "error"
        base_result["error"] = f"Post-processing failed: {exc}"
        return base_result

    base_result.update({
        "name": contact["name"],
        "email": contact["email"],
        "phone": contact["phone"],
        "skills": skills,
        "sections": sections,
        "raw_text": raw_text,
        "parsing_status": "success",
        "error": None,
    })
    return base_result


def process_all_resumes(resumes_dir: str = DEFAULT_RESUMES_DIR,
                         output_path: str = DEFAULT_OUTPUT_PATH) -> list:
    """
    Process every *.pdf in resumes_dir. One bad PDF never stops the batch.
    Writes the results list to output_path (creating parent dirs as needed) and
    also returns it.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf_paths = sorted(glob.glob(os.path.join(resumes_dir, "*.pdf")))

    results = []
    for pdf_path in pdf_paths:
        result = process_resume(pdf_path)
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ---------------------------------------------------------------------------
# 8. JOB DESCRIPTION PROCESSING
# ---------------------------------------------------------------------------

def process_job_description(path: str = DEFAULT_JD_PATH) -> dict:
    """
    Read and lightly process the job description: raw text + deterministic
    skill list, for the ranking team to consume alongside resume data.
    """
    if not os.path.isfile(path):
        return {"raw_text": "", "skills": [], "error": f"File not found: {path}"}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except Exception as exc:
        return {"raw_text": "", "skills": [], "error": f"Unable to read JD: {exc}"}

    cleaned = clean_text(raw_text)
    skills = extract_skills(cleaned)

    return {"raw_text": raw_text, "skills": skills, "error": None}


# ---------------------------------------------------------------------------
# CLI entry point: `python -m src.parser.pdf_parser`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = process_all_resumes()
    success = sum(1 for r in all_results if r["parsing_status"] == "success")
    print(f"Processed {len(all_results)} resumes -> {success} succeeded")
    for r in all_results:
        print(f"  {r['filename']}: {r['parsing_status']}"
              + (f" ({r['error']})" if r["error"] else ""))
    print(f"Output written to: {DEFAULT_OUTPUT_PATH}")
