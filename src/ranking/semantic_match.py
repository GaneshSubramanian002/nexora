import re


def tokenize(text):
    """Convert text into normalized words."""
    if not text:
        return set()

    return set(
        re.findall(
            r"[a-zA-Z0-9+#.]+",
            text.lower()
        )
    )


def semantic_match(resume, job_description):
    """
    Calculate a lightweight semantic-style similarity score.

    Uses the resume's summary, projects, experience and skills
    against the job description.
    """

    resume_text = " ".join([
        resume.get("sections", {}).get("summary", ""),
        resume.get("sections", {}).get("experience", ""),
        resume.get("sections", {}).get("projects", ""),
        " ".join(resume.get("skills", []))
    ])

    resume_words = tokenize(resume_text)
    jd_words = tokenize(job_description)

    if not jd_words:
        return 0.0

    common_words = resume_words.intersection(jd_words)

    score = (len(common_words) / len(jd_words)) * 100

    return round(min(score, 100.0), 2)
