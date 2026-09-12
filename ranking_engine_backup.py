"""
ranking_engine.py

Smart Shortlisting Engine
-------------------------
Hybrid Resume Ranking Engine using:

1. Explicit required/preferred skill matching
2. TF-IDF + cosine similarity
3. Sentence Transformer semantic similarity
4. Experience/project semantic similarity

Main functions:

    rank_candidates(jd_text, resumes)
    compare_candidates(candidate_a, candidate_b)

Input format:

jd_text = "Job description..."

resumes = [
    {
        "name": "Candidate A",
        "raw_text": "...",
        "skills": ["React", "Node.js", "MongoDB"]
    },
    ...
]

Output:
A list of ranked candidate dictionaries.
"""


import re
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Final scoring weights
SEMANTIC_WEIGHT = 0.40
REQUIRED_SKILL_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.20
EXPERIENCE_WEIGHT = 0.10


# ============================================================
# CONTROLLED SKILL VOCABULARY
# ============================================================
#
# We intentionally use a controlled vocabulary rather than
# building a complicated NLP skill extractor.
#
# The vocabulary can easily be expanded if the JD mentions
# another technology.
#
# Each skill has aliases so that:
#
# "Node" == "Node.js"
# "ReactJS" == "React"
# "RESTful API" == "REST API"
#
# ============================================================

SKILL_ALIASES = {

    # Frontend
    "React": [
        "react",
        "reactjs",
        "react.js"
    ],

    "Angular": [
        "angular",
        "angularjs"
    ],

    "Vue.js": [
        "vue",
        "vuejs",
        "vue.js"
    ],

    "JavaScript": [
        "javascript",
        "js",
        "ecmascript"
    ],

    "TypeScript": [
        "typescript",
        "ts"
    ],

    "HTML": [
        "html",
        "html5"
    ],

    "CSS": [
        "css",
        "css3"
    ],

    "Tailwind CSS": [
        "tailwind",
        "tailwindcss",
        "tailwind css"
    ],

    # Backend
    "Node.js": [
        "node",
        "nodejs",
        "node.js"
    ],

    "Express": [
        "express",
        "expressjs",
        "express.js"
    ],

    "Python": [
        "python"
    ],

    "Java": [
        "java"
    ],

    "C++": [
        "c++",
        "cpp"
    ],

    "C": [
        " c ",
        "c programming",
        "c language"
    ],

    "Django": [
        "django"
    ],

    "Flask": [
        "flask"
    ],

    "Spring Boot": [
        "spring boot",
        "springboot"
    ],

    # Databases
    "MongoDB": [
        "mongodb",
        "mongo db",
        "mongo"
    ],

    "MySQL": [
        "mysql"
    ],

    "PostgreSQL": [
        "postgresql",
        "postgres"
    ],

    "Redis": [
        "redis"
    ],

    "SQL": [
        "sql",
        "structured query language"
    ],

    # APIs / architecture
    "REST API": [
        "rest api",
        "restful api",
        "restful apis",
        "rest services",
        "rest service",
        "restful services"
    ],

    "GraphQL": [
        "graphql",
        "graph ql"
    ],

    "Microservices": [
        "microservices",
        "microservice",
        "micro services"
    ],

    # DevOps / Cloud
    "Git": [
        "git",
        "github",
        "gitlab",
        "version control"
    ],

    "Docker": [
        "docker",
        "containerization",
        "containerisation"
    ],

    "Kubernetes": [
        "kubernetes",
        "k8s"
    ],

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Azure": [
        "azure",
        "microsoft azure"
    ],

    "Google Cloud": [
        "gcp",
        "google cloud",
        "google cloud platform"
    ],

    # Other common technologies
    "Firebase": [
        "firebase"
    ],

    "Next.js": [
        "next.js",
        "nextjs",
        "next js"
    ],

    "Redux": [
        "redux"
    ],

    "Jest": [
        "jest"
    ],

    "Selenium": [
        "selenium"
    ],

    "Linux": [
        "linux"
    ],

    "CI/CD": [
        "ci/cd",
        "cicd",
        "continuous integration",
        "continuous deployment"
    ]
}


# ============================================================
# MODEL LOADING
# ============================================================

_model = None


def get_model():
    """
    Load the Sentence Transformer model only once.

    This prevents the model from being loaded repeatedly
    every time rank_candidates() is called.
    """

    global _model

    if _model is None:
        print("Loading semantic model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Semantic model loaded.")

    return _model


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:
    """
    Convert input to clean lowercase text.

    Handles:
    - None
    - non-string values
    - extra whitespace
    """

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    # Replace common separators with spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill: Any) -> str:
    """
    Normalize a skill name.
    """

    return normalize_text(skill)


# ============================================================
# SKILL MATCHING
# ============================================================

def skill_present(text: str, aliases: List[str]) -> bool:
    """
    Check whether one of the aliases occurs in the text.

    Uses word-boundary matching where possible.
    """

    normalized = normalize_text(text)

    for alias in aliases:

        alias = normalize_text(alias)

        if not alias:
            continue

        # Some aliases intentionally contain spaces or symbols.
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"

        if re.search(pattern, normalized):
            return True

    return False


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from text using the controlled vocabulary.

    This is intentionally simple and reliable.
    """

    text = normalize_text(text)

    found_skills = []

    for skill, aliases in SKILL_ALIASES.items():

        if skill_present(text, aliases):
            found_skills.append(skill)

    return found_skills


def normalize_resume_skills(skills: Any) -> List[str]:
    """
    Convert the resume's optional 'skills' field into
    canonical skill names.

    Example:

        ["ReactJS", "Node", "MongoDB"]

    becomes:

        ["React", "Node.js", "MongoDB"]
    """

    if skills is None:
        return []

    if isinstance(skills, str):
        skills = [skills]

    if not isinstance(skills, list):
        return []

    canonical_skills = []

    for supplied_skill in skills:

        supplied_skill_normalized = normalize_skill(supplied_skill)

        if not supplied_skill_normalized:
            continue

        matched = False

        for canonical_skill, aliases in SKILL_ALIASES.items():

            if (
                supplied_skill_normalized == normalize_skill(canonical_skill)
                or supplied_skill_normalized in [
                    normalize_skill(alias) for alias in aliases
                ]
            ):
                if canonical_skill not in canonical_skills:
                    canonical_skills.append(canonical_skill)

                matched = True
                break

        # If it is not in our controlled vocabulary, preserve it.
        if not matched:
            canonical_skills.append(str(supplied_skill).strip())

    return canonical_skills


# ============================================================
# JD SKILL EXTRACTION
# ============================================================

def identify_jd_skills(jd_text: str):
    """
    Identify required and preferred skills from the JD.

    The function first extracts all recognizable skills.

    It then tries to identify whether a skill is in a section
    containing words such as:

        required
        must have
        mandatory
        essential

    or:

        preferred
        nice to have
        bonus
        plus
        desirable

    If section detection is uncertain, recognized skills are
    treated as required.

    This intentionally favors recall and reliability for the
    hackathon.
    """

    jd = normalize_text(jd_text)

    all_skills = extract_skills_from_text(jd)

    required = []
    preferred = []

    # Look for explicit "required" sections.
    required_section = ""

    required_patterns = [
        r"required skills?(.*?)(?=preferred|nice to have|bonus|desirable|responsibilit|qualification|$)",
        r"must have(.*?)(?=preferred|nice to have|bonus|desirable|responsibilit|qualification|$)",
        r"mandatory(.*?)(?=preferred|nice to have|bonus|desirable|responsibilit|qualification|$)",
        r"essential(.*?)(?=preferred|nice to have|bonus|desirable|responsibilit|qualification|$)"
    ]

    for pattern in required_patterns:

        match = re.search(pattern, jd, flags=re.IGNORECASE)

        if match:
            required_section += " " + match.group(1)

    # Look for preferred sections.
    preferred_section = ""

    preferred_patterns = [
        r"preferred skills?(.*?)(?=required|must have|mandatory|essential|responsibilit|qualification|$)",
        r"nice to have(.*?)(?=required|must have|mandatory|essential|responsibilit|qualification|$)",
        r"bonus(.*?)(?=required|must have|mandatory|essential|responsibilit|qualification|$)",
        r"desirable(.*?)(?=required|must have|mandatory|essential|responsibilit|qualification|$)"
    ]

    for pattern in preferred_patterns:

        match = re.search(pattern, jd, flags=re.IGNORECASE)

        if match:
            preferred_section += " " + match.group(1)

    required_section = normalize_text(required_section)
    preferred_section = normalize_text(preferred_section)

    # If explicit required section exists, use it.
    if required_section:
        for skill in all_skills:

            aliases = SKILL_ALIASES.get(skill, [skill])

            if skill_present(required_section, aliases):
                if skill not in required:
                    required.append(skill)

    # If explicit preferred section exists, use it.
    if preferred_section:
        for skill in all_skills:

            aliases = SKILL_ALIASES.get(skill, [skill])

            if skill_present(preferred_section, aliases):
                if skill not in preferred:
                    preferred.append(skill)

    # Skills that are explicitly preferred should not also
    # be counted as required.
    required_set = set(required)

    preferred = [
        skill for skill in preferred
        if skill not in required_set
    ]

    # If automatic section detection didn't find enough required
    # skills, treat remaining skills as required.
    #
    # This is important for JDs written like:
    #
    # "Looking for React, Node.js, MongoDB and Git."
    #
    # where there may be no "Required Skills" heading.
    if not required:
        required = list(all_skills)

    else:
        # Skills not explicitly marked preferred are generally
        # treated as required.
        for skill in all_skills:

            if skill not in required and skill not in preferred:
                required.append(skill)

    return required, preferred


# ============================================================
# EXPLICIT REQUIRED SKILL SCORE
# ============================================================

def calculate_required_skill_score(
    resume_text: str,
    resume_skills: List[str],
    required_skills: List[str]
):
    """
    Calculate:

        matched required skills / total required skills

    Returned on a 0-100 scale.
    """

    if not required_skills:
        return 100.0, [], []

    resume_text = normalize_text(resume_text)

    canonical_resume_skills = set(
        normalize_skill(skill)
        for skill in resume_skills
    )

    matched = []
    missing = []

    for skill in required_skills:

        skill_normalized = normalize_skill(skill)

        # Check supplied skills field first.
        explicit_match = skill_normalized in canonical_resume_skills

        # Also check raw resume text.
        text_match = skill_present(
            resume_text,
            SKILL_ALIASES.get(skill, [skill])
        )

        if explicit_match or text_match:
            matched.append(skill)
        else:
            missing.append(skill)

    score = (len(matched) / len(required_skills)) * 100.0

    return score, matched, missing


# ============================================================
# PREFERRED SKILL SCORE
# ============================================================

def calculate_preferred_skill_score(
    resume_text: str,
    resume_skills: List[str],
    preferred_skills: List[str]
):
    """
    Calculate preferred skill coverage.

    Returned on a 0-100 scale.
    """

    if not preferred_skills:
        return 100.0, []

    resume_text = normalize_text(resume_text)

    canonical_resume_skills = set(
        normalize_skill(skill)
        for skill in resume_skills
    )

    matched = []

    for skill in preferred_skills:

        explicit_match = normalize_skill(skill) in canonical_resume_skills

        text_match = skill_present(
            resume_text,
            SKILL_ALIASES.get(skill, [skill])
        )

        if explicit_match or text_match:
            matched.append(skill)

    score = (len(matched) / len(preferred_skills)) * 100.0

    return score, matched


# ============================================================
# TF-IDF KEYWORD MATCHING
# ============================================================

def calculate_keyword_scores(
    jd_text: str,
    resume_texts: List[str]
):
    """
    Calculate TF-IDF cosine similarity between the JD and
    every resume.

    The raw cosine similarity is converted to a 0-100 score.

    Example:

        0.82 -> 82
    """

    documents = [normalize_text(jd_text)] + [
        normalize_text(text)
        for text in resume_texts
    ]

    # Handle empty input safely.
    if not any(documents):
        return [0.0] * len(resume_texts)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:
        matrix = vectorizer.fit_transform(documents)

    except ValueError:
        # This happens if every document is empty.
        return [0.0] * len(resume_texts)

    jd_vector = matrix[0]

    resume_vectors = matrix[1:]

    similarities = cosine_similarity(
        jd_vector,
        resume_vectors
    )[0]

    scores = [
        round(float(max(0.0, min(1.0, score))) * 100.0, 2)
        for score in similarities
    ]

    return scores


# ============================================================
# SEMANTIC MATCHING
# ============================================================

def calculate_semantic_scores(
    jd_text: str,
    resume_texts: List[str]
):
    """
    Calculate semantic similarity using:

        all-MiniLM-L6-v2

    JD is encoded once.

    All resumes are encoded in a batch.

    Returns 0-100 scores.
    """

    model = get_model()

    jd_text_clean = normalize_text(jd_text)

    clean_resumes = [
        normalize_text(text)
        for text in resume_texts
    ]

    if not jd_text_clean:
        return [0.0] * len(resume_texts)

    if not clean_resumes:
        return []

    jd_embedding = model.encode(
        [jd_text_clean],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    resume_embeddings = model.encode(
        clean_resumes,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = np.dot(
        resume_embeddings,
        jd_embedding[0]
    )

    scores = []

    for similarity in similarities:

        # Cosine similarity is theoretically [-1, 1].
        # Sentence-transformer similarities for this use case
        # are normally positive.
        #
        # Convert [-1, 1] to [0, 100].
        normalized = (float(similarity) + 1.0) / 2.0

        normalized = max(
            0.0,
            min(1.0, normalized)
        )

        scores.append(
            round(normalized * 100.0, 2)
        )

    return scores


# ============================================================
# EXPERIENCE / PROJECT EXTRACTION
# ============================================================

def extract_experience_and_projects(resume_text: str) -> str:
    """
    Extract a lightweight representation of experience/projects.

    This is intentionally NOT a complicated resume parser.

    We search for common resume headings and capture text around
    those sections.

    If section detection fails, the full resume is returned.

    This makes the engine robust to different resume formats.
    """

    text = normalize_text(resume_text)

    if not text:
        return ""

    section_patterns = [
        "experience",
        "work experience",
        "professional experience",
        "internship",
        "internships",
        "projects",
        "academic projects",
        "personal projects",
        "work history"
    ]

    # Find positions of likely headings.
    positions = []

    for heading in section_patterns:

        pattern = r"\b" + re.escape(heading) + r"\b"

        match = re.search(pattern, text)

        if match:
            positions.append(match.start())

    if not positions:
        return text

    # Take a generous portion starting from the earliest
    # experience/project heading.
    start = min(positions)

    experience_text = text[start:]

    # Avoid extremely large inputs.
    return experience_text[:6000]


# ============================================================
# EXPERIENCE SEMANTIC SCORE
# ============================================================

def calculate_experience_scores(
    jd_text: str,
    resume_texts: List[str]
):
    """
    Compare the JD against the experience/project portions
    of each resume using semantic similarity.
    """

    model = get_model()

    jd_clean = normalize_text(jd_text)

    experience_texts = [
        extract_experience_and_projects(text)
        for text in resume_texts
    ]

    if not jd_clean:
        return [0.0] * len(resume_texts)

    if not experience_texts:
        return []

    jd_embedding = model.encode(
        [jd_clean],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    experience_embeddings = model.encode(
        experience_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = np.dot(
        experience_embeddings,
        jd_embedding[0]
    )

    scores = []

    for similarity in similarities:

        normalized = (float(similarity) + 1.0) / 2.0

        normalized = max(
            0.0,
            min(1.0, normalized)
        )

        scores.append(
            round(normalized * 100.0, 2)
        )

    return scores


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    semantic_score: float,
    required_skill_score: float,
    keyword_score: float,
    experience_score: float
):
    """
    Calculate the final hybrid score.

    Formula:

        40% semantic
        30% required skills
        20% keyword
        10% experience
    """

    score = (
        SEMANTIC_WEIGHT * semantic_score
        + REQUIRED_SKILL_WEIGHT * required_skill_score
        + KEYWORD_WEIGHT * keyword_score
        + EXPERIENCE_WEIGHT * experience_score
    )

    return round(
        max(0.0, min(100.0, score)),
        2
    )


# ============================================================
# MAIN RANKING FUNCTION
# ============================================================

def rank_candidates(
    jd_text: str,
    resumes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Main function used by the UI/backend.

    Parameters
    ----------
    jd_text : str
        Complete job description.

    resumes : list of dict
        Example:

        [
            {
                "name": "Candidate A",
                "raw_text": "...",
                "skills": [
                    "React",
                    "Node.js",
                    "MongoDB"
                ]
            }
        ]

    Returns
    -------
    list of dictionaries sorted by final_score descending.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not isinstance(jd_text, str):
        raise ValueError("jd_text must be a string.")

    if not isinstance(resumes, list):
        raise ValueError("resumes must be a list.")

    if len(resumes) == 0:
        return []

    # --------------------------------------------------------
    # Identify JD skills
    # --------------------------------------------------------

    required_skills, preferred_skills = identify_jd_skills(
        jd_text
    )

    # --------------------------------------------------------
    # Prepare resume text
    # --------------------------------------------------------

    resume_texts = []

    prepared_resumes = []

    for index, resume in enumerate(resumes):

        if not isinstance(resume, dict):
            raise ValueError(
                f"Resume at index {index} must be a dictionary."
            )

        name = resume.get(
            "name",
            f"Candidate {index + 1}"
        )

        raw_text = resume.get(
            "raw_text",
            ""
        )

        skills = resume.get(
            "skills",
            []
        )

        # If raw_text is missing, use skills as text.
        if not raw_text:
            if isinstance(skills, list):
                raw_text = " ".join(
                    str(skill) for skill in skills
                )
            else:
                raw_text = str(skills)

        raw_text = str(raw_text)

        normalized_skills = normalize_resume_skills(
            skills
        )

        # If the skills field is empty, automatically extract
        # skills from the resume.
        extracted_skills = extract_skills_from_text(
            raw_text
        )

        combined_skills = []

        for skill in normalized_skills + extracted_skills:

            if skill not in combined_skills:
                combined_skills.append(skill)

        resume_texts.append(raw_text)

        prepared_resumes.append({
            "name": str(name),
            "raw_text": raw_text,
            "skills": combined_skills
        })

    # --------------------------------------------------------
    # Calculate all scoring components
    # --------------------------------------------------------

    print("\nCalculating TF-IDF keyword scores...")

    keyword_scores = calculate_keyword_scores(
        jd_text,
        resume_texts
    )

    print("Calculating semantic scores...")

    semantic_scores = calculate_semantic_scores(
        jd_text,
        resume_texts
    )

    print("Calculating experience/project scores...")

    experience_scores = calculate_experience_scores(
        jd_text,
        resume_texts
    )

    # --------------------------------------------------------
    # Build candidate results
    # --------------------------------------------------------

    results = []

    for i, resume in enumerate(prepared_resumes):

        required_score, matched_required, missing_required = (
            calculate_required_skill_score(
                resume["raw_text"],
                resume["skills"],
                required_skills
            )
        )

        preferred_score, matched_preferred = (
            calculate_preferred_skill_score(
                resume["raw_text"],
                resume["skills"],
                preferred_skills
            )
        )

        semantic_score = semantic_scores[i]
        keyword_score = keyword_scores[i]
        experience_score = experience_scores[i]

        final_score = calculate_final_score(
            semantic_score=semantic_score,
            required_skill_score=required_score,
            keyword_score=keyword_score,
            experience_score=experience_score
        )

        result = {

            # Basic information
            "rank": 0,
            "name": resume["name"],

            # Main scoring components
            "semantic_score": round(
                semantic_score,
                2
            ),

            "keyword_score": round(
                keyword_score,
                2
            ),

            "required_skill_score": round(
                required_score,
                2
            ),

            "preferred_skill_score": round(
                preferred_score,
                2
            ),

            "experience_score": round(
                experience_score,
                2
            ),

            # Final hybrid score
            "final_score": round(
                final_score,
                2
            ),

            # Explainability
            "matched_required_skills": matched_required,

            "missing_required_skills": missing_required,

            "matched_preferred_skills": matched_preferred,

            # Useful metadata for debugging/demo
            "required_skills": required_skills,

            "preferred_skills": preferred_skills
        }

        results.append(result)

    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    results.sort(
        key=lambda candidate: candidate["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Assign ranks
    # --------------------------------------------------------

    for rank, candidate in enumerate(
        results,
        start=1
    ):
        candidate["rank"] = rank

    return results


# ============================================================
# CANDIDATE COMPARISON
# ============================================================

def compare_candidates(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two candidate result dictionaries produced by
    rank_candidates().

    Example:

        comparison = compare_candidates(
            results[0],
            results[1]
        )

    Returns numerical differences and missing skills.

    Positive difference means Candidate A has the higher score.
    """

    if not isinstance(candidate_a, dict):
        raise ValueError(
            "candidate_a must be a dictionary."
        )

    if not isinstance(candidate_b, dict):
        raise ValueError(
            "candidate_b must be a dictionary."
        )

    def get_score(candidate, key):
        try:
            return float(candidate.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    semantic_difference = (
        get_score(candidate_a, "semantic_score")
        - get_score(candidate_b, "semantic_score")
    )

    keyword_difference = (
        get_score(candidate_a, "keyword_score")
        - get_score(candidate_b, "keyword_score")
    )

    required_difference = (
        get_score(candidate_a, "required_skill_score")
        - get_score(candidate_b, "required_skill_score")
    )

    experience_difference = (
        get_score(candidate_a, "experience_score")
        - get_score(candidate_b, "experience_score")
    )

    final_difference = (
        get_score(candidate_a, "final_score")
        - get_score(candidate_b, "final_score")
    )

    return {

        "candidate_a": candidate_a.get(
            "name",
            "Candidate A"
        ),

        "candidate_b": candidate_b.get(
            "name",
            "Candidate B"
        ),

        "semantic_difference": round(
            semantic_difference,
            2
        ),

        "keyword_difference": round(
            keyword_difference,
            2
        ),

        "required_skill_difference": round(
            required_difference,
            2
        ),

        "experience_difference": round(
            experience_difference,
            2
        ),

        "final_score_difference": round(
            final_difference,
            2
        ),

        "a_missing_skills": candidate_a.get(
            "missing_required_skills",
            []
        ),

        "b_missing_skills": candidate_b.get(
            "missing_required_skills",
            []
        ),

        "a_matched_required_skills": candidate_a.get(
            "matched_required_skills",
            []
        ),

        "b_matched_required_skills": candidate_b.get(
            "matched_required_skills",
            []
        )
    }


# ============================================================
# TOP CANDIDATES HELPER
# ============================================================

def get_top_candidates(
    ranked_candidates: List[Dict[str, Any]],
    n: int = 3
) -> List[Dict[str, Any]]:
    """
    Return the top N candidates.
    """

    if n <= 0:
        return []

    return ranked_candidates[:n]


# ============================================================
# PRINT RANKING TABLE
# ============================================================

def print_ranking_table(
    results: List[Dict[str, Any]]
):
    """
    Print a simple terminal-friendly ranking table.

    Useful for testing the 18 resumes during the hackathon.
    """

    if not results:
        print("No candidates found.")
        return

    print("\n")
    print("=" * 100)
    print("SMART SHORTLISTING ENGINE - CANDIDATE RANKING")
    print("=" * 100)

    header = (
        f"{'Rank':<6}"
        f"{'Candidate':<25}"
        f"{'Semantic':<12}"
        f"{'Required':<12}"
        f"{'Keyword':<12}"
        f"{'Experience':<12}"
        f"{'FINAL':<10}"
    )

    print(header)
    print("-" * 100)

    for candidate in results:

        print(
            f"{candidate['rank']:<6}"
            f"{candidate['name'][:23]:<25}"
            f"{candidate['semantic_score']:<12.2f}"
            f"{candidate['required_skill_score']:<12.2f}"
            f"{candidate['keyword_score']:<12.2f}"
            f"{candidate['experience_score']:<12.2f}"
            f"{candidate['final_score']:<10.2f}"
        )

    print("=" * 100)


# ============================================================
# PRINT CANDIDATE DETAILS
# ============================================================

def print_candidate_details(
    candidate: Dict[str, Any]
):
    """
    Print detailed explainability information for one candidate.
    """

    print("\n")
    print("=" * 70)
    print(
        f"RANK #{candidate.get('rank')} - "
        f"{candidate.get('name')}"
    )
    print("=" * 70)

    print(
        f"Final Score       : "
        f"{candidate.get('final_score', 0):.2f}"
    )

    print(
        f"Semantic Score    : "
        f"{candidate.get('semantic_score', 0):.2f}"
    )

    print(
        f"Required Skills   : "
        f"{candidate.get('required_skill_score', 0):.2f}"
    )

    print(
        f"Keyword Score     : "
        f"{candidate.get('keyword_score', 0):.2f}"
    )

    print(
        f"Experience Score  : "
        f"{candidate.get('experience_score', 0):.2f}"
    )

    print(
        f"Preferred Skills  : "
        f"{candidate.get('preferred_skill_score', 0):.2f}"
    )

    print("\nMatched Required Skills:")

    matched = candidate.get(
        "matched_required_skills",
        []
    )

    if matched:
        for skill in matched:
            print(f"  + {skill}")
    else:
        print("  None")

    print("\nMissing Required Skills:")

    missing = candidate.get(
        "missing_required_skills",
        []
    )

    if missing:
        for skill in missing:
            print(f"  - {skill}")
    else:
        print("  None")

    print("\nMatched Preferred Skills:")

    preferred = candidate.get(
        "matched_preferred_skills",
        []
    )

    if preferred:
        for skill in preferred:
            print(f"  + {skill}")
    else:
        print("  None")

    print("=" * 70)


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example Job Description
    # --------------------------------------------------------

    jd_text = """
    We are looking for a Full Stack Developer.

    Required Skills:
    React
    Node.js
    Express
    MongoDB
    REST API
    Git

    Preferred Skills:
    TypeScript
    Docker
    AWS

    The candidate should have experience developing web
    applications, backend REST services, APIs and scalable
    full stack applications.
    """

    # --------------------------------------------------------
    # Example resumes
    #
    # Person 2 can replace this list with the actual 18 resumes.
    # --------------------------------------------------------

    resumes = [
    {
        "name": "Perfect Candidate",
        "raw_text": """
        Full stack developer with experience in React,
        Node.js, Express, MongoDB, REST APIs and Git.
        Built multiple web applications.
        """,
        "skills": [
            "React",
            "Node.js",
            "Express",
            "MongoDB",
            "REST API",
            "Git"
        ]
    },

    {
        "name": "Weak Candidate",
        "raw_text": """
        Python developer with experience in Django
        and PostgreSQL.
        Built backend applications.
        """,
        "skills": [
            "Python",
            "Django",
            "PostgreSQL"
        ]
    }
]
    # --------------------------------------------------------
    # RUN RANKING
    # --------------------------------------------------------

    results = rank_candidates(
        jd_text,
        resumes
    )

    # --------------------------------------------------------
    # PRINT COMPLETE RANKING
    # --------------------------------------------------------

    print_ranking_table(results)

    # --------------------------------------------------------
    # Print top 3
    # --------------------------------------------------------

    print("\nTOP 3 CANDIDATES")
    print("-" * 50)

    top_candidates = get_top_candidates(
        results,
        3
    )

    for candidate in top_candidates:

        print(
            f"#{candidate['rank']} "
            f"{candidate['name']} "
            f"-> {candidate['final_score']:.2f}"
        )

    # --------------------------------------------------------
    # Detailed information for Rank 1
    # --------------------------------------------------------

    if results:
        print_candidate_details(
            results[0]
        )

    # --------------------------------------------------------
    # Compare Rank 1 vs Rank 2
    # --------------------------------------------------------

    if len(results) >= 2:

        comparison = compare_candidates(
            results[0],
            results[1]
        )

        print("\n")
        print("=" * 70)
        print("CANDIDATE COMPARISON")
        print("=" * 70)

        print(
            f"{comparison['candidate_a']} vs "
            f"{comparison['candidate_b']}"
        )

        print(
            f"Semantic Difference       : "
            f"{comparison['semantic_difference']:.2f}"
        )

        print(
            f"Keyword Difference        : "
            f"{comparison['keyword_difference']:.2f}"
        )

        print(
            f"Required Skill Difference : "
            f"{comparison['required_skill_difference']:.2f}"
        )

        print(
            f"Experience Difference     : "
            f"{comparison['experience_difference']:.2f}"
        )

        print(
            f"Final Score Difference    : "
            f"{comparison['final_score_difference']:.2f}"
        )

        print("\nA Missing Skills:")
        print(
            comparison["a_missing_skills"]
        )

        print("\nB Missing Skills:")
        print(
            comparison["b_missing_skills"]
        )

        print("=" * 70)