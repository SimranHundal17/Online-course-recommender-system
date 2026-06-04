from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Online Course Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
PROCESSED_COURSES_PATH = ROOT / "data" / "processed" / "processed_udemy_courses.csv"


SECTIONS = [
    "Project Overview",
    "Course Recommendation",
    "Algorithm Comparison",
    "Evaluation Results",
    "Synthetic Learner Profiles",
    "Graphs and Results",
    "Limitations",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f9fc;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #667085;
            --line: #d9e2ef;
            --blue: #2563eb;
            --green: #059669;
            --amber: #d97706;
            --purple: #7c3aed;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero {
            background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 62%, #6d28d9 100%);
            border-radius: 18px;
            padding: 2rem;
            color: #ffffff;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 50px rgba(37, 99, 235, 0.22);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
            line-height: 1.05;
            font-weight: 800;
        }

        .hero p {
            margin: 0.75rem 0 0 0;
            max-width: 900px;
            color: rgba(255,255,255,0.88);
            font-size: 1rem;
            line-height: 1.55;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.85rem;
        }

        .card h3 {
            margin: 0 0 0.4rem 0;
            color: var(--ink);
        }

        .card p, .card li {
            color: var(--muted);
            line-height: 1.55;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            background: #eef4ff;
            color: #1d4ed8;
            border: 1px solid #c7d7fe;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .note {
            border-left: 4px solid var(--blue);
            background: #eff6ff;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            color: #1e3a8a;
            margin: 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Online Course Recommender System</h1>
            <p>
                A final project dashboard for exploring content-based course recommendations,
                comparing TF-IDF and KNN recommenders, reviewing evaluation results, and
                demonstrating synthetic learner personas for interface personalization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_courses() -> pd.DataFrame:
    if not PROCESSED_COURSES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_COURSES_PATH)


def format_number(value: int | float) -> str:
    return f"{int(value):,}"


def render_dataset_metrics(courses: pd.DataFrame) -> None:
    subjects = courses["subject"].nunique() if "subject" in courses else 0
    levels = courses["level"].nunique() if "level" in courses else 0
    cols = st.columns(4)
    cols[0].metric("Courses", format_number(len(courses)))
    cols[1].metric("Subjects", subjects)
    cols[2].metric("Course Levels", levels)
    cols[3].metric("Feature Columns", len(courses.columns))


def render_course_details(course: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="card">
            <h3>{course["course_title"]}</h3>
            <span class="badge">{course["subject"]}</span>
            <span class="badge">{course["level"]}</span>
            <span class="badge">{format_number(course["num_subscribers"])} subscribers</span>
            <span class="badge">{format_number(course["num_reviews"])} reviews</span>
            <p>
                <strong>Price:</strong> {course["price"]} |
                <strong>Duration:</strong> {course["content_duration"]} hours
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_overview_section() -> None:
    st.header("Project Overview")
    courses = load_courses()
    if not courses.empty:
        render_dataset_metrics(courses)
        st.subheader("Available Dataset Coverage")
        subject_badges = " ".join(
            f'<span class="badge">{subject}</span>'
            for subject in sorted(courses["subject"].dropna().unique())
        )
        level_badges = " ".join(
            f'<span class="badge">{level}</span>'
            for level in sorted(courses["level"].dropna().unique())
        )
        st.markdown(f"**Subjects:**<br>{subject_badges}", unsafe_allow_html=True)
        st.markdown(f"**Levels:**<br>{level_badges}", unsafe_allow_html=True)
    else:
        st.warning(f"Processed dataset not found at `{PROCESSED_COURSES_PATH}`.")

    cols = st.columns(3)
    with cols[0]:
        render_card("Recommendation Goal", "Suggest relevant Udemy-style courses using course metadata and content similarity.")
    with cols[1]:
        render_card("Models Compared", "TF-IDF with cosine similarity is compared against a KNN content-based recommender.")
    with cols[2]:
        render_card("Evaluation Focus", "The project evaluates relevance, overlap, runtime, and feature ablation instead of supervised metrics.")


def course_recommendation_section() -> None:
    st.header("Course Recommendation")
    courses = load_courses()
    if courses.empty:
        st.warning(f"Processed dataset not found at `{PROCESSED_COURSES_PATH}`.")
        return

    render_dataset_metrics(courses)

    st.subheader("Select a Course")
    selected_title = st.selectbox(
        "Course title",
        courses["course_title"].sort_values().tolist(),
        index=0,
    )
    selected_course = courses[courses["course_title"] == selected_title].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Selected Course Details")
        render_course_details(selected_course)
    with right:
        st.subheader("Dataset Filters Available")
        st.write("These filters will be used with recommendation logic in the next stage.")
        st.write("**Available subjects:**")
        st.write(", ".join(sorted(courses["subject"].dropna().unique())))
        st.write("**Available levels:**")
        st.write(", ".join(sorted(courses["level"].dropna().unique())))


def algorithm_comparison_section() -> None:
    st.header("Algorithm Comparison")
    st.info("Stage 1 layout placeholder. Saved TF-IDF vs KNN example results will be connected in later stages.")


def evaluation_results_section() -> None:
    st.header("Evaluation Results")
    st.info("Stage 1 layout placeholder. Evaluation CSV files and graphs will be connected in later stages.")


def synthetic_profiles_section() -> None:
    st.header("Synthetic Learner Profiles")
    st.markdown(
        """
        <div class="note">
            Synthetic learner profiles are for demonstration only. This is not collaborative filtering.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Stage 1 layout placeholder. Synthetic persona data will be connected in later stages.")


def graphs_section() -> None:
    st.header("Graphs and Results")
    st.info("Stage 1 layout placeholder. Exported graph files and summaries will be connected in later stages.")


def limitations_section() -> None:
    st.header("Limitations")
    st.markdown(
        """
        - The dataset does not contain real user interaction data.
        - Recommendation evaluation is qualitative and comparative rather than based on supervised labels.
        - Synthetic personas are only for demonstration and do not represent real users.
        - The dashboard is a final project interface, not a production recommendation service.
        """
    )


def main() -> None:
    inject_styles()
    with st.sidebar:
        st.title("Navigation")
        section = st.radio("Go to", SECTIONS)
        st.divider()
        st.caption("Online Course Recommender System")
        st.caption("Content-based TF-IDF and KNN comparison")

    render_header()

    if section == "Project Overview":
        project_overview_section()
    elif section == "Course Recommendation":
        course_recommendation_section()
    elif section == "Algorithm Comparison":
        algorithm_comparison_section()
    elif section == "Evaluation Results":
        evaluation_results_section()
    elif section == "Synthetic Learner Profiles":
        synthetic_profiles_section()
    elif section == "Graphs and Results":
        graphs_section()
    elif section == "Limitations":
        limitations_section()


if __name__ == "__main__":
    main()
