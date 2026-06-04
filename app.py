from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
COURSES = [
    {
        "title": "Python for Data Science and Machine Learning",
        "category": "Web Development",
        "level": "Beginner",
        "subscribers": "48.2K",
        "rating": 4.8,
        "description": "A practical introduction to Python, data analysis, and applied machine learning workflows.",
    },
    {
        "title": "Complete Financial Analyst Training",
        "category": "Business Finance",
        "level": "Intermediate",
        "subscribers": "31.7K",
        "rating": 4.7,
        "description": "A structured finance course covering valuation, Excel modeling, and investment analysis.",
    },
    {
        "title": "Modern Web Design with HTML, CSS, and JavaScript",
        "category": "Web Development",
        "level": "Beginner",
        "subscribers": "62.4K",
        "rating": 4.6,
        "description": "A project-based web design course focused on responsive layouts and interactive interfaces.",
    },
    {
        "title": "Advanced Guitar Techniques Masterclass",
        "category": "Musical Instruments",
        "level": "Expert",
        "subscribers": "12.9K",
        "rating": 4.9,
        "description": "An advanced music course for improving speed, expression, improvisation, and performance.",
    },
    {
        "title": "Graphic Design Bootcamp: From Basics to Portfolio",
        "category": "Graphic Design",
        "level": "Intermediate",
        "subscribers": "27.5K",
        "rating": 4.7,
        "description": "A visual design course covering typography, layout, branding, and portfolio projects.",
    },
]

RECOMMENDATIONS = [
    {
        "title": "Applied Python Projects for Beginners",
        "category": "Web Development",
        "level": "Beginner",
        "similarity": 0.94,
        "popularity": 91,
        "tfidf": 0.94,
        "knn": 0.87,
        "explanation": "Strong overlap in Python, projects, and beginner-friendly technical learning.",
    },
    {
        "title": "Data Analysis with Pandas and Visualization",
        "category": "Web Development",
        "level": "Intermediate",
        "similarity": 0.90,
        "popularity": 88,
        "tfidf": 0.91,
        "knn": 0.84,
        "explanation": "Matches core data analysis terms and attracts a similar learner profile.",
    },
    {
        "title": "Machine Learning Foundations with Scikit-Learn",
        "category": "Web Development",
        "level": "Intermediate",
        "similarity": 0.87,
        "popularity": 86,
        "tfidf": 0.88,
        "knn": 0.82,
        "explanation": "Recommended because of close topic similarity and strong engagement metrics.",
    },
    {
        "title": "Excel Analytics for Business Decisions",
        "category": "Business Finance",
        "level": "Intermediate",
        "similarity": 0.79,
        "popularity": 84,
        "tfidf": 0.76,
        "knn": 0.86,
        "explanation": "KNN favors its similar workload, popularity, price range, and learner difficulty level.",
    },
    {
        "title": "Responsive Web Interfaces in Practice",
        "category": "Web Development",
        "level": "Beginner",
        "similarity": 0.82,
        "popularity": 89,
        "tfidf": 0.84,
        "knn": 0.80,
        "explanation": "A close content-based match with related programming and interface-building vocabulary.",
    },
    {
        "title": "Portfolio Design for Digital Creators",
        "category": "Graphic Design",
        "level": "Beginner",
        "similarity": 0.73,
        "popularity": 78,
        "tfidf": 0.70,
        "knn": 0.81,
        "explanation": "Adds diversity while staying close to creative technical course characteristics.",
    },
]

SYNTHETIC_USERS_PATH = Path("data/synthetic_users.csv")
PROCESSED_COURSES_PATH = Path("data/processed/processed_udemy_courses.csv")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #08111f;
        --panel: rgba(16, 28, 48, 0.82);
        --panel-strong: rgba(20, 35, 59, 0.94);
        --border: rgba(148, 163, 184, 0.18);
        --text: #e5edf8;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --accent-2: #22c55e;
        --accent-3: #f59e0b;
        --accent-4: #a78bfa;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.20), transparent 34rem),
            radial-gradient(circle at 85% 5%, rgba(167, 139, 250, 0.18), transparent 30rem),
            linear-gradient(135deg, #07111f 0%, #0b1424 45%, #101827 100%);
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 2.75rem;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none;
    }

    div[data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        top: 0.65rem !important;
        left: 0.75rem !important;
        z-index: 999999 !important;
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 12px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
    }

    div[data-testid="collapsedControl"] svg {
        color: #f8fafc !important;
        stroke: #f8fafc !important;
    }

    div[data-testid="stAppViewContainer"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1324 0%, #111c31 100%);
        border-right: 1px solid var(--border);
        width: 320px !important;
        min-width: 320px !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #dbe7f7 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        border-radius: 12px !important;
        min-height: 46px;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border-radius: 999px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] span {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(135deg, #38bdf8, #6366f1) !important;
        border: 0 !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        min-height: 48px;
        box-shadow: 0 16px 34px rgba(37, 99, 235, 0.28);
    }

    section[data-testid="stSidebar"] button[kind="secondary"] p {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[role="slider"] {
        background: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.86));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
    }

    div[data-testid="stMetricLabel"] {
        color: #c6d3e4 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800;
        font-size: 2.25rem;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 800;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(14, 165, 233, 0.95), rgba(79, 70, 229, 0.92)),
            linear-gradient(135deg, rgba(34, 197, 94, 0.16), transparent);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 24px;
        padding: clamp(1.7rem, 3vw, 2.4rem);
        margin-bottom: 1.4rem;
        box-shadow: 0 24px 70px rgba(2, 8, 23, 0.40);
    }

    .hero h1 {
        font-size: clamp(2.1rem, 3.5vw, 4rem);
        line-height: 1.02;
        letter-spacing: 0;
        margin: 0;
        color: #ffffff;
    }

    .hero h3 {
        margin: 0.75rem 0 0.8rem 0;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 600;
    }

    .hero p {
        max-width: 860px;
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.03rem;
        margin: 0;
    }

    .section-title {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 800;
        margin: 1rem 0 0.75rem 0;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
    }

    .course-overview {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.78));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.5rem;
        min-height: 245px;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .course-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.15;
        margin-bottom: 0.65rem;
    }

    .description {
        color: #b8c5d6;
        line-height: 1.55;
        margin-top: 0.75rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.8rem 0;
    }

    .pill {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 999px;
        color: #dbeafe;
        background: rgba(15, 23, 42, 0.56);
        padding: 0.42rem 0.72rem;
        font-size: 0.83rem;
        font-weight: 700;
    }

    .rec-card {
        background: linear-gradient(150deg, rgba(15, 23, 42, 0.96), rgba(21, 32, 55, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    .rec-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.50);
        box-shadow: 0 22px 46px rgba(14, 165, 233, 0.16);
    }

    .rec-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }

    .rank {
        min-width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-4));
        color: white;
        font-weight: 800;
    }

    .rec-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 1.06rem;
        line-height: 1.25;
    }

    .score-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.9rem 0;
    }

    .score-box {
        border-radius: 14px;
        background: rgba(2, 6, 23, 0.36);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 0.7rem;
    }

    .score-box span {
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .score-box strong {
        color: #f8fafc;
        font-size: 1.18rem;
    }

    .comparison-card {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem;
        min-height: 215px;
    }

    .comparison-card h3 {
        margin: 0 0 0.5rem 0;
        color: #f8fafc;
    }

    .comparison-card p, .comparison-card li {
        color: #b8c5d6;
        line-height: 1.5;
    }

    .persona-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.78));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.5rem;
        min-height: 245px;
    }

    .notice {
        background: rgba(14, 165, 233, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #dbeafe;
        line-height: 1.5;
        margin: 0.8rem 0 1rem 0;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
        padding-top: 1.2rem;
        font-weight: 700;
    }

    @media (max-width: 1100px) {
        section[data-testid="stSidebar"] {
            width: 285px !important;
            min-width: 285px !important;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_synthetic_users() -> pd.DataFrame:
    if not SYNTHETIC_USERS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "user_id",
                "persona_name",
                "interests",
                "preferred_level",
                "preferred_subject",
            ]
        )
    return pd.read_csv(SYNTHETIC_USERS_PATH)


@st.cache_data
def load_processed_courses() -> pd.DataFrame:
    if not PROCESSED_COURSES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_COURSES_PATH)


def get_selected_course(title: str) -> dict:
    return next(course for course in COURSES if course["title"] == title)


def filter_recommendations(levels: list[str], limit: int, algorithm: str) -> list[dict]:
    filtered = [course for course in RECOMMENDATIONS if course["level"] in levels]
    score_key = "tfidf" if algorithm.startswith("TF-IDF") else "knn"
    filtered = sorted(filtered, key=lambda item: item[score_key], reverse=True)
    return filtered[:limit]


def render_recommendation_card(course: dict, rank: int, algorithm: str) -> None:
    score_key = "tfidf" if algorithm.startswith("TF-IDF") else "knn"
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-top">
                <div>
                    <div class="eyebrow">{course["category"]} · {course["level"]}</div>
                    <div class="rec-title">{course["title"]}</div>
                </div>
                <div class="rank">#{rank}</div>
            </div>
            <div class="score-grid">
                <div class="score-box">
                    <span>Similarity</span>
                    <strong>{course[score_key]:.2f}</strong>
                </div>
                <div class="score-box">
                    <span>Popularity</span>
                    <strong>{course["popularity"]}%</strong>
                </div>
            </div>
            <p class="description">{course["explanation"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_persona_recommendations(persona: pd.Series, limit: int) -> list[dict]:
    courses_df = load_processed_courses()
    if courses_df.empty:
        return []

    interests = str(persona["interests"]).lower()
    interest_terms = [
        term.strip()
        for chunk in interests.split(";")
        for term in chunk.split()
        if len(term.strip()) > 2
    ]
    preferred_subject = str(persona["preferred_subject"]).lower()
    preferred_level = str(persona["preferred_level"]).lower()

    scored_courses = courses_df.copy()
    searchable_text = (
        scored_courses["course_title"].fillna("")
        + " "
        + scored_courses["subject"].fillna("")
        + " "
        + scored_courses["level"].fillna("")
        + " "
        + scored_courses["combined_features"].fillna("")
    ).str.lower()

    scored_courses["match_score"] = searchable_text.apply(
        lambda text: sum(1 for term in interest_terms if term in text)
    )
    scored_courses["subject_match"] = (
        scored_courses["subject"].fillna("").str.lower() == preferred_subject
    ).astype(int)
    scored_courses["level_match"] = (
        scored_courses["level"].fillna("").str.lower() == preferred_level
    ).astype(int)
    scored_courses["persona_score"] = (
        scored_courses["match_score"]
        + scored_courses["subject_match"] * 2
        + scored_courses["level_match"]
    )

    recommendations_df = scored_courses.sort_values(
        ["persona_score", "match_score", "num_subscribers"],
        ascending=[False, False, False],
    ).head(limit)

    recommendations = []
    for _, row in recommendations_df.iterrows():
        recommendations.append(
            {
                "title": row["course_title"],
                "category": row["subject"],
                "level": row["level"],
                "similarity": min(row["persona_score"] / 8, 0.99),
                "popularity": min(int(row["num_subscribers"]) // 1000 + 55, 99),
                "tfidf": min(row["persona_score"] / 8, 0.99),
                "knn": min(row["persona_score"] / 8, 0.99),
                "explanation": (
                    "Simple demo match based on persona interests, preferred subject, "
                    "and preferred level."
                ),
            }
        )
    return recommendations


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Recommendation Controls")
    st.caption("Prototype interface using realistic mock outputs.")

    dashboard_mode = st.radio(
        "Dashboard Mode",
        ["Course-Based Recommendation", "Learner Persona Recommendation"],
        index=0,
    )

    if dashboard_mode == "Course-Based Recommendation":
        algorithm = st.selectbox(
            "Algorithm",
            ["TF-IDF + Cosine Similarity", "KNN Recommendation"],
            index=0,
        )
        selected_course_title = st.selectbox(
            "Selected Course",
            [course["title"] for course in COURSES],
            index=0,
        )
        recommendation_count = st.slider("Recommendation Count", 3, 6, 5)
        levels = st.multiselect(
            "Difficulty Filter",
            ["Beginner", "Intermediate", "Expert"],
            default=["Beginner", "Intermediate", "Expert"],
        )
        run_recommendation = st.button("Generate Recommendations", use_container_width=True)
    else:
        algorithm = "Synthetic Persona Matching"
        recommendation_count = st.slider("Recommendation Count", 3, 6, 5)
        synthetic_users = load_synthetic_users()
        persona_names = synthetic_users["persona_name"].tolist()
        selected_persona_name = st.selectbox(
            "Learner Persona",
            persona_names,
            index=0,
            disabled=synthetic_users.empty,
        )
        levels = ["Beginner", "Intermediate", "Expert"]
        selected_course_title = COURSES[0]["title"]
        run_recommendation = st.button("Generate Persona Recommendations", use_container_width=True)

    st.divider()
    st.markdown("### Demo Status")
    st.success("Interface ready")
    if dashboard_mode == "Course-Based Recommendation":
        st.caption("Algorithms will be connected in later notebooks.")
    else:
        st.caption("Synthetic learner profiles are for interface personalization demos only.")

if not levels:
    levels = ["Beginner", "Intermediate", "Expert"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hero_subtitle = (
    "Comparison of TF-IDF and KNN Recommendation Algorithms"
    if dashboard_mode == "Course-Based Recommendation"
    else "Synthetic Learner Profile Demonstration"
)
hero_description = (
    "A modern dashboard prototype for exploring course recommendations using content similarity, "
    "course metadata, and mock algorithm performance insights before final model integration."
    if dashboard_mode == "Course-Based Recommendation"
    else "A demonstration mode showing how synthetic learner personas can support interface personalization. "
    "These are sample profiles only and are not collaborative filtering users."
)

st.markdown(
    f"""
    <div class="hero">
        <h1>Online Course Recommender System</h1>
        <h3>{hero_subtitle}</h3>
        <p>{hero_description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_persona = None
if dashboard_mode == "Course-Based Recommendation":
    selected_course = get_selected_course(selected_course_title)
    recommendations = filter_recommendations(levels, recommendation_count, algorithm)
    active_score = "tfidf" if algorithm.startswith("TF-IDF") else "knn"
else:
    synthetic_users = load_synthetic_users()
    if synthetic_users.empty:
        selected_persona = pd.Series(
            {
                "persona_name": "No personas available",
                "interests": "Add data/synthetic_users.csv to enable this mode.",
                "preferred_level": "N/A",
                "preferred_subject": "N/A",
            }
        )
        recommendations = []
    else:
        selected_persona = synthetic_users[
            synthetic_users["persona_name"] == selected_persona_name
        ].iloc[0]
        recommendations = build_persona_recommendations(selected_persona, recommendation_count)
    active_score = "tfidf"

avg_similarity = (
    sum(course[active_score] for course in recommendations) / len(recommendations)
    if recommendations
    else 0
)
avg_popularity = (
    sum(course["popularity"] for course in recommendations) / len(recommendations)
    if recommendations
    else 0
)
diversity = len({course["category"] for course in recommendations}) / max(len(recommendations), 1)

if run_recommendation:
    if dashboard_mode == "Course-Based Recommendation":
        st.toast(f"{algorithm} recommendations generated for the selected course.")
    else:
        st.toast("Synthetic learner persona recommendations generated.")


# ---------------------------------------------------------------------------
# Metrics dashboard
# ---------------------------------------------------------------------------
metric_cols = st.columns(5)
metric_cols[0].metric("Similarity Score", f"{avg_similarity:.2f}", "+4.8%")
runtime_label = (
    "42 ms"
    if algorithm.startswith("TF-IDF")
    else "67 ms"
    if algorithm.startswith("KNN")
    else "demo"
)
metric_cols[1].metric("Runtime", runtime_label, "-12 ms" if runtime_label != "demo" else "sample")
metric_cols[2].metric("Diversity", f"{diversity * 100:.0f}%", "+9%")
metric_cols[3].metric("Confidence", f"{min(avg_popularity + 6, 99):.0f}%", "+3%")
metric_cols[4].metric("Recommendations", len(recommendations), "ready")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
overview_col, rec_col = st.columns([0.95, 1.35], gap="large")

with overview_col:
    if dashboard_mode == "Course-Based Recommendation":
        st.markdown('<div class="section-title">Selected Course Overview</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="course-overview">
                <div class="eyebrow">Input Course</div>
                <div class="course-title">{selected_course["title"]}</div>
                <div class="pill-row">
                    <span class="pill">{selected_course["category"]}</span>
                    <span class="pill">{selected_course["level"]}</span>
                    <span class="pill">{selected_course["subscribers"]} subscribers</span>
                    <span class="pill">{selected_course["rating"]:.1f} rating</span>
                </div>
                <p class="description">{selected_course["description"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-title">Learner Persona Overview</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="persona-card">
                <div class="eyebrow">Synthetic Learner Profile</div>
                <div class="course-title">{escape(str(selected_persona["persona_name"]))}</div>
                <div class="pill-row">
                    <span class="pill">{escape(str(selected_persona["preferred_subject"]))}</span>
                    <span class="pill">{escape(str(selected_persona["preferred_level"]))}</span>
                    <span class="pill">demo profile</span>
                </div>
                <p class="description"><strong>Interests:</strong> {escape(str(selected_persona["interests"]))}</p>
            </div>
            <div class="notice">
                This mode uses synthetic learner personas for interface demonstration only.
                It is not collaborative filtering and does not represent real users.
            </div>
            """,
            unsafe_allow_html=True,
        )

    comparison_title = (
        "Algorithm Comparison"
        if dashboard_mode == "Course-Based Recommendation"
        else "Persona Matching Notes"
    )
    st.markdown(f'<div class="section-title">{comparison_title}</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        if dashboard_mode == "Course-Based Recommendation":
            st.markdown(
                """
                <div class="comparison-card">
                    <h3>TF-IDF Recommendations</h3>
                    <p>Prioritizes textual overlap between course titles, subjects, and difficulty levels.</p>
                    <ul>
                        <li>Fast and explainable</li>
                        <li>Strong for topic matching</li>
                        <li>Less aware of numeric metadata</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="comparison-card">
                    <h3>Synthetic Profile Demo</h3>
                    <p>Personas describe sample learner interests, preferred level, and preferred subject.</p>
                    <ul>
                        <li>Not real user data</li>
                        <li>Not collaborative filtering</li>
                        <li>Designed for UI personalization demos</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        if dashboard_mode == "Course-Based Recommendation":
            st.markdown(
                """
                <div class="comparison-card">
                    <h3>KNN Recommendations</h3>
                    <p>Compares courses using metadata patterns such as duration, lectures, popularity, and price.</p>
                    <ul>
                        <li>Useful for structured similarity</li>
                        <li>Sensitive to feature scaling</li>
                        <li>Can balance course workload and demand</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="comparison-card">
                    <h3>Simple Matching Logic</h3>
                    <p>Recommendations are ranked by matching persona interests with course titles, subject, level, and combined features.</p>
                    <ul>
                        <li>Transparent demonstration method</li>
                        <li>Uses processed Udemy course metadata</li>
                        <li>Ready to connect to final model outputs later</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

with rec_col:
    recommendation_title = (
        "Recommended Courses"
        if dashboard_mode == "Course-Based Recommendation"
        else "Persona-Based Demo Recommendations"
    )
    st.markdown(f'<div class="section-title">{recommendation_title}</div>', unsafe_allow_html=True)
    for index, recommendation in enumerate(recommendations, start=1):
        render_recommendation_card(recommendation, index, algorithm)


# ---------------------------------------------------------------------------
# Visualization section
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Recommendation Analytics</div>', unsafe_allow_html=True)

chart_col_1, chart_col_2, chart_col_3 = st.columns(3, gap="large")

score_df = pd.DataFrame(
    (
        {
            "Course": [f"Rank {index}" for index in range(1, len(recommendations) + 1)],
            "TF-IDF": [course["tfidf"] for course in recommendations],
            "KNN": [course["knn"] for course in recommendations],
        }
        if dashboard_mode == "Course-Based Recommendation"
        else {
            "Course": [f"Rank {index}" for index in range(1, len(recommendations) + 1)],
            "Persona Match": [course["tfidf"] for course in recommendations],
            "Popularity": [course["popularity"] / 100 for course in recommendations],
        }
    )
).set_index("Course")

category_df = (
    pd.DataFrame(recommendations)
    .groupby("category")
    .size()
    .rename("Recommended Courses")
    .to_frame()
)

performance_df = pd.DataFrame(
    (
        {
            "Metric": ["Precision", "Runtime", "Diversity", "Explainability"],
            "TF-IDF": [86, 94, 72, 91],
            "KNN": [82, 78, 84, 76],
        }
        if dashboard_mode == "Course-Based Recommendation"
        else {
            "Metric": ["Interest Match", "Subject Fit", "Level Fit", "Demo Clarity"],
            "Persona Matching": [88, 92, 84, 96],
        }
    )
).set_index("Metric")

with chart_col_1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Score Comparison" if dashboard_mode == "Course-Based Recommendation" else "Persona Match Scores")
    st.line_chart(score_df, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Category Distribution")
    st.bar_chart(category_df, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Algorithm Performance" if dashboard_mode == "Course-Based Recommendation" else "Demo Profile Signals")
    st.bar_chart(performance_df, height=260)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Smart Information Systems Course Project
    </div>
    """,
    unsafe_allow_html=True,
)
