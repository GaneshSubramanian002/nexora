# Nexora Resume Ranking Hackathon

Local, deterministic resume ranking system. No LLMs, no external APIs, no
cloud services — everything runs on-machine.

## Team ownership

| Area | Owner | Path |
|---|---|---|
| PDF/resume parsing | Person 2 | `src/parser/` |
| Ranking | Person 1 | `src/ranking/` |
| Explainability | Person 4 | `src/explainability/` |
| UI | Person 3 | `src/ui/` |

## Parser (`src/parser/pdf_parser.py`)

### Setup

```bash
pip install -r requirements.txt
```

### Input

Drop resume PDFs into `data/resumes/` (any filenames, `*.pdf` is picked up
automatically). Put the job description text at `data/job_description.txt`.

### Running

```bash
python -m src.parser.pdf_parser
```

This processes every PDF in `data/resumes/` and writes the results to
`data/output/resumes.json` (created automatically).

### Using it programmatically

```python
from src.parser.pdf_parser import process_resume, process_all_resumes, process_job_description

# Single resume
result = process_resume("data/resumes/resume_001.pdf")

# All resumes -> also writes data/output/resumes.json
results = process_all_resumes()

# Job description
jd = process_job_description("data/job_description.txt")
```

### Output shape (per resume)

```json
{
  "id": "resume_001",
  "filename": "resume_001.pdf",
  "name": "Jane Doe",
  "email": "jane.doe@example.com",
  "phone": "+91 98765 43210",
  "skills": ["AWS", "Python", "React"],
  "sections": {
    "summary": "...",
    "skills": "...",
    "experience": "...",
    "education": "...",
    "projects": "...",
    "certifications": "...",
    "achievements": "..."
  },
  "raw_text": "...",
  "parsing_status": "success",
  "error": null
}
```

`parsing_status` is one of `success`, `no_text` (PDF opened but had no
extractable text), or `error` (see `error` for details). One bad PDF never
stops the rest of the batch.

Any field that couldn't be confidently identified (e.g. `name`, `email`,
`phone`) is `null` rather than guessed.

### For Person 1 (ranking) / Person 4 (explainability)

Import the JSON at `data/output/resumes.json`, or call
`process_all_resumes()` directly. Every resume dict always has all of the
fields above, even when a section/field is empty — safe to index without
`.get()` guards.

### Tests

```bash
python -m pytest tests/test_parser.py -v
```
