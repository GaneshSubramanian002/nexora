from flask import Flask, render_template, jsonify, request
import json
import os
from werkzeug.utils import secure_filename

from src.ranking.ranker import rank_candidates
from src.parser.pdf_parser import process_all_resumes


app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESUMES_DIR = os.path.join(
    BASE_DIR,
    "data",
    "resumes"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "output",
    "resumes.json"
)

ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(RESUMES_DIR, exist_ok=True)
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# LOAD RESUMES
# --------------------------------------------------

def load_resumes():

    if not os.path.exists(OUTPUT_FILE):
        return []

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# CHECK FILE TYPE
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# UPLOAD RESUME
# --------------------------------------------------

@app.route(
    "/api/upload-resume",
    methods=["POST"]
)
def upload_resume():

    if "resume" not in request.files:

        return jsonify({
            "error": "No resume file uploaded."
        }), 400


    file = request.files["resume"]


    if file.filename == "":

        return jsonify({
            "error": "No file selected."
        }), 400


    if not allowed_file(file.filename):

        return jsonify({
            "error": "Only PDF resumes are supported."
        }), 400


    filename = secure_filename(
        file.filename
    )


    save_path = os.path.join(
        RESUMES_DIR,
        filename
    )


    file.save(save_path)


    # Re-process all resumes
    processed = process_all_resumes(
        resumes_dir=RESUMES_DIR,
        output_path=OUTPUT_FILE
    )


    uploaded_resume = next(
        (
            resume
            for resume in processed
            if resume.get("filename") == filename
        ),
        None
    )


    if uploaded_resume is None:

        return jsonify({
            "error": "Resume was saved but could not be processed."
        }), 500


    if uploaded_resume.get(
        "parsing_status"
    ) != "success":

        return jsonify({
            "error":
                uploaded_resume.get(
                    "error",
                    "Resume parsing failed."
                )
        }), 500


    return jsonify({

        "message":
            "Resume uploaded successfully.",

        "resume":
            uploaded_resume,

        "total_candidates":
            len(processed)

    })


# --------------------------------------------------
# RANK CANDIDATES
# --------------------------------------------------

@app.route("/api/candidates")
def get_candidates():

    query = request.args.get(
        "query",
        ""
    ).strip()


    if not query:

        return jsonify({
            "error":
                "Please provide a job description or search query."
        }), 400


    resumes = load_resumes()


    if not resumes:

        return jsonify({
            "error":
                "No resumes found."
        }), 404


    skills_list = [

        "Python",
        "Java",
        "C",
        "C++",
        "C#",
        "JavaScript",
        "TypeScript",
        "HTML",
        "CSS",
        "React",
        "Angular",
        "Vue",
        "Node.js",
        "Express",
        "Django",
        "Flask",
        "FastAPI",
        "Spring",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Redis",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",
        "REST API",
        "GraphQL",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Pandas",
        "NumPy",
        "scikit-learn",
        "Linux",
        "Firebase",
        "Tailwind",
        "Bootstrap"

    ]


    required_skills = [

        skill

        for skill in skills_list

        if skill.lower()
        in query.lower()

    ]


    ranked = rank_candidates(

        resumes,

        query,

        required_skills

    )


    return jsonify(ranked)


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
