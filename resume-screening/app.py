"""
SmartHire AI - Streamlit Resume Screening System

This single-file app supports:
1) Job creation
2) Candidate applications through shareable links
3) AI scoring (TF-IDF / optional BERT)
4) Hiring decision workflow (Pending / Shortlisted / Rejected)
"""

import json
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path("data_store.json")


# -------------------------
# Data save/load functions
# JSON = simple file-based storage (no database)
# -------------------------
def load_data() -> Dict[str, Dict]:
    """Load data from JSON file into memory."""
    if not DATA_FILE.exists():
        return {"jobs": {}}
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "jobs" not in data:
            data["jobs"] = {}
        return data
    except Exception:
        return {"jobs": {}}


def save_data(data: Dict[str, Dict]) -> None:
    """Persist in-memory data to JSON file."""
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------------
# Text and PDF helper functions
# Normalize text = make text lowercase/clean for fair comparison
# -------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_skills(skills_text: str) -> List[str]:
    """Convert comma-separated skills into a clean unique list."""
    skills = [s.strip().lower() for s in skills_text.split(",") if s.strip()]
    # Remove duplicate skills but keep original order
    return list(dict.fromkeys(skills))


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF upload."""
    pdf_bytes = uploaded_file.read()
    text_chunks = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


# =====================================================================
# CORE AI PART - How Resume Matching Works (Simple Explanation)
# =====================================================================
# GOAL: Compare job description vs resume text and give a match score (0-100).
#
# We use 2 methods (user can choose):
#
# METHOD 1 - TF-IDF (default, no extra install needed):
#   - Step 1: Convert both texts into "word importance" numbers.
#             (TF-IDF = which words appear often in this doc but rare elsewhere)
#   - Step 2: Put those numbers in a list → this is a "vector".
#   - Step 3: Cosine similarity = angle between 2 vectors.
#             Same meaning → vectors point same way → high score.
#             Different meaning → vectors point away → low score.
#   - Step 4: Score 0–1 is scaled to 0–100.
#
# METHOD 2 - BERT (optional, needs sentence-transformers): (Deep-Learning is used by this method)
#   - Step 1: BERT = a pre-trained neural network that understands sentence meaning.
#   - Step 2: Model turns each text into a list of numbers (embedding).
#   - Step 3: Cosine similarity on embeddings → how similar in meaning.
#   - Step 4: Score 0–1 is scaled to 0–100.
#
# WHY COSINE SIMILARITY?
#   It measures how "aligned" two vectors are. 1 = same direction (very similar).
#   0 = perpendicular. -1 = opposite. We use it for both TF-IDF and BERT.
# =====================================================================

def compute_tfidf_score(job_description: str, resume_text: str) -> float:
    """TF-IDF: convert texts to vectors, then cosine similarity. Returns 0-100."""
    docs = [job_description or "", resume_text or ""]
    vectorizer = TfidfVectorizer(stop_words="english")  # ignore words like 'the','is'
    tfidf_matrix = vectorizer.fit_transform(docs)       # each doc -> vector
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


@st.cache_resource(show_spinner=False)
def load_bert_model():
    """Load BERT model once and reuse. all-MiniLM-L6-v2 = small, fast sentence model."""
    from sentence_transformers import SentenceTransformer  # optional dependency

    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_bert_score(job_description: str, resume_text: str) -> float:
    """BERT: encode texts to embeddings, then cosine similarity. Returns 0-100."""
    model = load_bert_model()
    embeddings = model.encode([job_description or "", resume_text or ""])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity) * 100, 2)


def compute_match_score(job_description: str, resume_text: str, method: str) -> float:
    """Pick TF-IDF or BERT based on user choice."""
    if method == "BERT (all-MiniLM-L6-v2)":
        return compute_bert_score(job_description, resume_text)
    return compute_tfidf_score(job_description, resume_text)


# -------------------------
# Skill/status UI helper functions
# Skill gap = required skills vs skills found in resume text
# -------------------------
def detect_skill_gap(required_skills: List[str], resume_text: str) -> Tuple[List[str], List[str]]:
    """Return matched and missing skills based on resume text."""
    resume_normalized = normalize_text(resume_text)
    matched = [skill for skill in required_skills if skill in resume_normalized]
    missing = [skill for skill in required_skills if skill not in resume_normalized]
    return matched, missing


def get_score_band(score: float) -> str:
    """Return score band label for visual feedback."""
    if score > 80:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def render_score_feedback(score: float) -> None:
    """Show color-coded score feedback."""
    band = get_score_band(score)
    msg = f"Screening score: **{score} / 100**"
    if band == "high":
        st.success(msg)
    elif band == "medium":
        st.warning(msg)
    else:
        st.error(msg)


def render_skill_tags(label: str, skills: List[str], color: str) -> None:
    """Render skills with simple colored tags."""
    st.markdown(f"**{label}**")
    if not skills:
        st.write("None")
        return

    tags = " ".join(
        [
            f"<span style='background:{color};padding:4px 10px;border-radius:14px;"
            f"margin:2px;display:inline-block;font-size:0.85rem;'>{skill}</span>"
            for skill in skills
        ]
    )
    st.markdown(tags, unsafe_allow_html=True)


def get_match_quality(score: float) -> str:
    """Map score to readable quality label."""
    if score > 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    return "Weak"


def render_status_badge(status: str) -> None:
    """Render status with consistent visual color."""
    normalized = (status or "Pending").strip().title()
    if normalized == "Shortlisted":
        st.success("Status: Shortlisted")
    elif normalized == "Rejected":
        st.error("Status: Rejected")
    else:
        st.warning("Status: Pending")


def ensure_candidate_status_defaults(data: Dict[str, Dict]) -> bool:
    """
    Ensure older candidate records have the new status field.
    Returns True when any update is made.
    """
    changed = False
    for job in data.get("jobs", {}).values():
        for candidate in job.get("candidates", []):
            if "status" not in candidate:
                candidate["status"] = "Pending"
                changed = True
    return changed


# -------------------------
# URL query helper functions
# Query params = values in URL like ?job_id=123
# -------------------------
def get_query_param(name: str) -> str:
    """Read query parameter safely from Streamlit."""
    try:
        val = st.query_params.get(name, "")
        if isinstance(val, list):
            return val[0] if val else ""
        return str(val)
    except Exception:
        # Backward compatibility for older Streamlit versions
        params = st.experimental_get_query_params()
        val = params.get(name, [""])
        return val[0] if isinstance(val, list) else str(val)


def set_query_param(name: str, value: str) -> None:
    """Set query parameter safely in Streamlit."""
    try:
        st.query_params[name] = value
    except Exception:
        st.experimental_set_query_params(**{name: value})


def build_application_link(job_id: str) -> str:
    base_url = st.secrets.get("APP_BASE_URL", "")
    if base_url:
        return f"{base_url}/?job_id={job_id}"
    return f"/?job_id={job_id}"   # fallback

# -------------------------
# Page 1: Create Job
# HR creates the role and gets a shareable application URL
# -------------------------
def create_job_page(data: Dict[str, Dict]) -> None:
    st.subheader("👨‍💼 Create Job")
    st.info("Define the role details, then share the generated application link with candidates.")
    st.markdown("**Workflow Step 1/4: Create Job**")
    st.markdown("---")
    st.markdown("### 🧾 Job Details")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
    with col2:
        experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=0, step=1)

    description = st.text_area("Job Description", height=180, placeholder="Describe responsibilities and qualifications...")
    skills_text = st.text_input(
        "Required Skills (comma separated)",
        placeholder="python, machine learning, sql, communication",
    )
    st.caption("Tip: Add 5-10 relevant skills for better matching quality.")

    if st.button("Create Job", type="primary", use_container_width=True):
        if not title.strip() or not description.strip() or not skills_text.strip():
            st.warning("Please fill Job Title, Job Description, and Required Skills.")
            return

        job_id = str(uuid.uuid4())
        skills = parse_skills(skills_text)
        # Store all job details in one dictionary object
        data["jobs"][job_id] = {
            "job_id": job_id,
            "title": title.strip(),
            "description": description.strip(),
            "skills": skills,
            "experience": int(experience),
            "candidates": [],
        }
        save_data(data)

        st.success("Job created successfully!")
        st.info("Next: Share this link with candidates.")
        application_link = build_application_link(job_id)
        st.markdown("#### 🔗 Application Link")
        st.code(application_link)
        st.caption("Share this link with candidates.")

        if st.button("Go to Apply Page", use_container_width=True):
            set_query_param("job_id", job_id)
            st.session_state["page"] = "Apply for Job"
            st.rerun()


# -------------------------
# Page 2: Apply for Job
# Candidate submits resume, app runs AI screening
# -------------------------
def apply_for_job_page(data: Dict[str, Dict]) -> None:
    st.subheader("👨‍💻 Apply for Job")
    st.info("Fill in your details and upload a PDF resume. The system will score your profile automatically.")
    st.markdown("**Workflow Step 2/4: Receive Applications**")

    jobs = data.get("jobs", {})
    query_job_id = get_query_param("job_id")
    selected_job_id = query_job_id if query_job_id in jobs else ""

    if not jobs:
        st.warning("No jobs available. Please create a job first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_job_id = st.selectbox(
            "Select Job ID",
            options=list(jobs.keys()),
            index=list(jobs.keys()).index(selected_job_id) if selected_job_id else 0,
        )
    with col2:
        if st.button("🔗 Generate Application Link", use_container_width=True):
            set_query_param("job_id", selected_job_id)
            link = build_application_link(selected_job_id)
            st.success("Application link generated.")
            st.code(link)
            st.caption("Share this link with candidates.")

    job = jobs[selected_job_id]
    st.markdown("---")
    st.markdown("### Selected Job")
    info_col1, info_col2, info_col3 = st.columns([2, 2, 1])
    with info_col1:
        st.write(f"**Title:** {job['title']}")
    with info_col2:
        st.write(f"**Required Skills:** {', '.join(job['skills'])}")
    with info_col3:
        st.write(f"**Experience:** {job['experience']} yrs")
    st.markdown("---")

    scoring_method = st.radio(
        "Scoring Method",
        options=["TF-IDF", "BERT (all-MiniLM-L6-v2)"],
        horizontal=True,
    )

    if scoring_method.startswith("BERT"):
        try:
            # Check if sentence-transformers is installed for BERT scoring
            __import__("sentence_transformers")
        except Exception:
            st.warning("BERT selected but `sentence-transformers` is not installed. Falling back to TF-IDF.")
            scoring_method = "TF-IDF"

    st.markdown("### Candidate Application Form")
    name_col, email_col = st.columns(2)
    with name_col:
        name = st.text_input("Name", placeholder="Your full name")
    with email_col:
        email = st.text_input("Email", placeholder="you@example.com")

    resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
    st.caption("Only text-based PDF resumes are supported for best extraction quality.")

    if st.button("Submit Application", type="primary", use_container_width=True):
        if not name.strip() or not email.strip() or resume_file is None:
            st.warning("Please provide name, email, and a PDF resume.")
            return

        with st.spinner("Analyzing resume and calculating match score..."):
            try:
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text:
                    st.warning("Could not extract text from this PDF. Please upload a text-based PDF.")
                    return
            except Exception as e:
                st.warning(f"Error reading PDF: {e}")
                return

            score = compute_match_score(job["description"], resume_text, scoring_method)
            matched, missing = detect_skill_gap(job["skills"], resume_text)

        # Save candidate data with AI score, skill gap, and current status
        candidate = {
            "name": name.strip(),
            "email": email.strip(),
            "resume_text": resume_text,
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "scoring_method": scoring_method,
            "status": "Pending",
        }
        job["candidates"].append(candidate)
        save_data(data)

        st.success("Application submitted successfully!")
        st.markdown("### Match Result")
        st.markdown(f"<h2 style='margin-top:0;'>🎯 {score} / 100</h2>", unsafe_allow_html=True)
        render_score_feedback(score)
        st.info(f"Match quality: **{get_match_quality(score)}**")
        st.info(f"Scoring method: **{scoring_method}**")
        with st.expander("Skill Gap Details", expanded=True):
            render_skill_tags("Matched Skills", matched, "#d1fae5")
            render_skill_tags("Missing Skills", missing, "#fee2e2")


# -------------------------
# Page 3: View Candidates
# HR sees insights, AI recommendation, and makes decisions
# -------------------------
def view_candidates_page(data: Dict[str, Dict]) -> None:
    st.subheader("📊 View Candidates")
    st.markdown("**Workflow Step 3/4: AI Analysis  |  Step 4/4: Hiring Decision**")
    jobs = data.get("jobs", {})

    if not jobs:
        st.warning("No jobs available yet.")
        return

    job_id = st.selectbox("Select Job ID", options=list(jobs.keys()))
    job = jobs[job_id]
    candidates = job.get("candidates", [])

    st.markdown("### 📊 Hiring Insights")
    total_candidates = len(candidates)
    avg_score = round(sum(c.get("score", 0) for c in candidates) / total_candidates, 2) if total_candidates else 0.0
    top_score = max((c.get("score", 0) for c in candidates), default=0.0)
    shortlisted_count = sum(1 for c in candidates if c.get("status", "Pending") == "Shortlisted")
    insight_cols = st.columns(4)
    insight_cols[0].metric("Total Candidates", total_candidates)
    insight_cols[1].metric("Average Score", avg_score)
    insight_cols[2].metric("Top Score", top_score)
    insight_cols[3].metric("Shortlisted", shortlisted_count)

    st.markdown("---")
    st.markdown("### 🎯 AI Recommended Candidate")
    if candidates:
        # AI recommendation rule: choose highest screening score
        best_candidate = max(candidates, key=lambda c: c.get("score", 0))
        matched_count = len(best_candidate.get("matched_skills", []))
        st.success(
            f"Recommended Candidate: **{best_candidate.get('name', 'Unknown')}**  |  "
            f"Score: **{best_candidate.get('score', 0)}**\n\n"
            f"Why selected: Best match based on skills and job description "
            f"with **{matched_count}** required skills matched."
        )
    else:
        st.info("No applications yet, so AI recommendation is not available.")

    st.markdown("---")
    st.markdown("### ✅ Shortlisted Candidates")
    shortlisted = [c for c in candidates if c.get("status", "Pending") == "Shortlisted"]
    if shortlisted:
        for idx, c in enumerate(sorted(shortlisted, key=lambda x: x.get("score", 0), reverse=True), start=1):
            st.success(f"#{idx} {c.get('name', 'Unknown')}  |  Score: {c.get('score', 0)}  |  {c.get('email', '')}")
    else:
        st.info("No shortlisted candidates yet.")

    st.markdown("---")
    st.markdown("### 🏆 Candidate Rankings")
    st.write(f"**Role:** {job['title']}")
    st.caption(f"{len(candidates)} candidate(s) applied")
    st.markdown("---")

    min_score = st.slider("Minimum score filter", min_value=0, max_value=100, value=0, step=1)
    st.caption("Move the slider to show only candidates above a minimum match score.")
    filtered = [c for c in candidates if c.get("score", 0) >= min_score]
    filtered_sorted = sorted(filtered, key=lambda c: c.get("score", 0), reverse=True)

    if not filtered_sorted:
        st.info("No candidates match the current filter.")
        return

    st.markdown("### Top Candidates")
    top_3 = filtered_sorted[:3]
    top_cols = st.columns(3)
    for idx, c in enumerate(top_3):
        with top_cols[idx]:
            rank = idx + 1
            st.markdown(
                f"#### Rank #{rank}\n"
                f"**{c.get('name', 'Unknown')}**\n\n"
                f"Score: **{c.get('score', 0)}**"
            )
            render_score_feedback(float(c.get("score", 0)))

    st.markdown("---")
    st.markdown("### Candidate Review Cards")
    for idx, c in enumerate(filtered_sorted):
        c_name = c.get("name", "Unknown")
        c_score = float(c.get("score", 0))
        c_status = c.get("status", "Pending")
        skill_match_summary = f"{len(c.get('matched_skills', []))}/{len(job.get('skills', []))} skills matched"

        card_title_cols = st.columns([2, 1, 1])
        with card_title_cols[0]:
            st.markdown(f"#### {c_name}")
            st.caption(f"{skill_match_summary} | Match quality: {get_match_quality(c_score)}")
        with card_title_cols[1]:
            st.markdown(f"### {c_score}")
        with card_title_cols[2]:
            render_status_badge(c_status)

        action_col1, action_col2, action_col3 = st.columns([1, 1, 3])
        with action_col1:
            if st.button("✅ Shortlist", key=f"shortlist_{job_id}_{idx}", use_container_width=True):
                # Persist hiring decision immediately in JSON
                c["status"] = "Shortlisted"
                save_data(data)
                st.success(f"{c_name} shortlisted.")
                st.rerun()
        with action_col2:
            if st.button("❌ Reject", key=f"reject_{job_id}_{idx}", use_container_width=True):
                # Persist hiring decision immediately in JSON
                c["status"] = "Rejected"
                save_data(data)
                st.error(f"{c_name} rejected.")
                st.rerun()
        with action_col3:
            if st.button("↩️ Set Pending", key=f"pending_{job_id}_{idx}", use_container_width=True):
                # Move candidate back to Pending (re-review state)
                c["status"] = "Pending"
                save_data(data)
                st.warning(f"{c_name} moved to pending.")
                st.rerun()

        with st.expander(f"{c_name} - Score: {c_score}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Email:** {c.get('email', '')}")
                st.write(f"**Method:** {c.get('scoring_method', 'TF-IDF')}")
                render_score_feedback(c_score)
                st.info(f"Match quality: **{get_match_quality(c_score)}**")
            with col2:
                render_skill_tags("Matched Skills", c.get("matched_skills", []), "#d1fae5")
                render_skill_tags("Missing Skills", c.get("missing_skills", []), "#fee2e2")

        st.markdown("---")

    st.markdown("### Candidate Table")
    table_data = [
        {
            "Name": c.get("name", ""),
            "Email": c.get("email", ""),
            "Score": c.get("score", 0),
            "Status": c.get("status", "Pending"),
        }
        for c in filtered_sorted
    ]
    st.dataframe(table_data, use_container_width=True)


# -------------------------
# App start and page routing
# Routing = deciding which page function to render
# -------------------------
def main() -> None:
    st.set_page_config(page_title="SmartHire AI", page_icon="🧠", layout="wide")
    st.title("🧠 SmartHire AI")
    st.caption("AI-powered Resume Screening & Candidate Ranking System")
    st.markdown("### How it Works")
    st.markdown(
        "1. Create a job\n"
        "2. Share application link\n"
        "3. Candidates upload resumes\n"
        "4. View ranked candidates"
    )
    st.markdown("---")

    data = load_data()
    if ensure_candidate_status_defaults(data):
        # Add default status to old candidate records (data migration)
        save_data(data)

    # Deep-link: if URL has job_id query param, open Apply page directly
    deep_link_job_id = get_query_param("job_id")
    if deep_link_job_id:
        st.session_state["page"] = "Apply for Job"

    page_labels = ["👨‍💼 Create Job", "👨‍💻 Apply for Job", "📊 View Candidates"]
    page_map = {
        "👨‍💼 Create Job": "Create Job",
        "👨‍💻 Apply for Job": "Apply for Job",
        "📊 View Candidates": "View Candidates",
    }
    default_page = st.session_state.get("page", "Create Job")
    reverse_map = {v: k for k, v in page_map.items()}
    default_label = reverse_map.get(default_page, "👨‍💼 Create Job")
    default_index = page_labels.index(default_label) if default_label in page_labels else 0

    with st.sidebar:
        st.header("Navigation")
        page_label = st.radio("Go to", options=page_labels, index=default_index)
        page = page_map[page_label]
        st.session_state["page"] = page
        st.success(f"Current Page: {page_label}")
        st.markdown("---")
        st.markdown("### Hiring Workflow")
        st.markdown(
            "1. **Create Job**\n"
            "2. **Receive Applications**\n"
            "3. **AI Analysis**\n"
            "4. **Hiring Decision**"
        )
        st.markdown("---")
        st.caption("Storage: in-memory + `data_store.json`")

    if page == "Create Job":
        create_job_page(data)
    elif page == "Apply for Job":
        apply_for_job_page(data)
    elif page == "View Candidates":
        view_candidates_page(data)

    st.markdown("---")
    st.caption("Project made by Anjali Jivani")


if __name__ == "__main__":
    main()
