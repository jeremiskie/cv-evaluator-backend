import re
import pdfplumber
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Load NLP Model
print("Loading NLP Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

OPEN_ROLES = {
    "Software Engineer": "Bachelor's degree in Computer Science. Responsible for building software applications, writing clean code, data structures, algorithms, Git, and REST APIs.",
    "Backend Developer": "Responsible for server-side logic, database management, Python, Java, Spring Boot, FastAPI, SQL, PostgreSQL, Docker, and API security.",
    "Frontend Developer": "Build user interfaces using HTML, CSS, JavaScript, TypeScript, React, Next.js, Tailwind CSS, Redux, and responsive design.",
    "Data Scientist": "Analyze complex datasets, build machine learning models using Python, Pandas, Scikit-Learn, PyTorch, statistics, and data cleaning.",
    "DevOps Engineer": "Manage cloud infrastructure, Docker containerization, Kubernetes, CI/CD pipelines, AWS, Terraform, and Linux administration.",
}


def extract_pdf_text(file_bytes) -> str:
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


@app.post("/evaluate")
async def evaluate_resume(
    target_role: str = Form(...), file: UploadFile = File(...)
):
    # Extract PDF Text
    raw_text = extract_pdf_text(file.file)
    cleaned_resume = re.sub(r"\s+", " ", raw_text).lower()

    # Encode Resume and Roles
    role_titles = list(OPEN_ROLES.keys())
    role_descriptions = [OPEN_ROLES[r].lower() for r in role_titles]

    resume_embedding = model.encode(cleaned_resume, convert_to_tensor=True)
    role_embeddings = model.encode(role_descriptions, convert_to_tensor=True)

    # Cosine Similarity
    scores_list = util.cos_sim(resume_embedding, role_embeddings)[0].tolist()
    scores = {
        title: round(score * 100, 1)
        for title, score in zip(role_titles, scores_list)
    }

    target_score = scores.get(target_role, 0.0)
    threshold = 45.0

    # Decision Logic
    is_accepted = target_score >= threshold
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_alt_role, best_alt_score = sorted_roles[0]

    recommendation = None
    if not is_accepted:
        if best_alt_role != target_role and best_alt_score >= threshold:
            recommendation = (
                f"Consider re-routing candidate to '{best_alt_role}'"
            )
        else:
            recommendation = (
                "No strong fit found among currently available open roles."
            )

    return {
        "candidate_summary": raw_text[:300] + "...",
        "target_role": target_role,
        "target_score": target_score,
        "is_accepted": is_accepted,
        "recommendation": recommendation,
        "all_scores": scores,
    }


# Run server: uvicorn main:app --reload