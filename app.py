from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


def get_course_index(courses: pd.DataFrame, title: str) -> int:
    matches = courses.index[courses["course_title"] == title].tolist()
    if not matches:
        raise ValueError(f"Course not found: {title}")
    return matches[0]


@st.cache_resource
def build_tfidf_recommender(courses: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(courses["combined_features"].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


@st.cache_resource
def build_knn_recommender(courses: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    text_matrix = vectorizer.fit_transform(courses["combined_features"].fillna(""))

    numeric_features = [
        column
        for column in ["num_subscribers", "num_reviews", "price", "content_duration"]
        if column in courses.columns
    ]

    if numeric_features:
        scaler = StandardScaler()
        numeric_matrix = scaler.fit_transform(courses[numeric_features].fillna(0))
        feature_matrix = hstack([text_matrix, numeric_matrix * 0.2])
    else:
        feature_matrix = text_matrix

    model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    model.fit(feature_matrix)
    return model, feature_matrix


def tfidf_recommendations(courses: pd.DataFrame, selected_title: str, n: int = 5) -> pd.DataFrame:
    _, _, similarity_matrix = build_tfidf_recommender(courses)
    selected_index = get_course_index(courses, selected_title)
    scores = list(enumerate(similarity_matrix[selected_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != selected_index][:n]

    result = courses.loc[
        [index for index, _ in scores],
        ["course_title", "subject", "level", "num_subscribers", "num_reviews", "price"],
    ].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["score"] = [round(float(score), 4) for _, score in scores]
    return result.reset_index(drop=True)


def knn_recommendations(courses: pd.DataFrame, selected_title: str, n: int = 5) -> pd.DataFrame:
    model, feature_matrix = build_knn_recommender(courses)
    selected_index = get_course_index(courses, selected_title)
    distances, indices = model.kneighbors(
        feature_matrix[selected_index],
        n_neighbors=n + 1,
    )

    rows = []
    for distance, index in zip(distances[0], indices[0]):
        if index == selected_index:
            continue
        row = courses.loc[index, ["course_title", "subject", "level", "num_subscribers", "num_reviews", "price"]].to_dict()
        row["distance"] = round(float(distance), 4)
        row["score"] = round(1 - float(distance), 4)
        rows.append(row)
        if len(rows) == n:
            break

    result = pd.DataFrame(rows)
    if not result.empty:
        result.insert(0, "rank", range(1, len(result) + 1))
    return result


def render_recommendation_table(title: str, recommendations: pd.DataFrame) -> None:
    st.subheader(title)
    if recommendations.empty:
        st.info("No recommendations available.")
        return
    st.dataframe(recommendations, use_container_width=True, hide_index=True)


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
    algorithm = st.radio(
        "Recommendation method",
        ["TF-IDF", "KNN", "Compare Both"],
        horizontal=True,
    )
    selected_course = courses[courses["course_title"] == selected_title].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Selected Course Details")
        render_course_details(selected_course)
    with right:
        st.subheader("Recommendation Setup")
        st.write("Recommendations are generated from the processed dataset using the project models.")
        st.write("**TF-IDF:** text similarity from `combined_features`.")
        st.write("**KNN:** text similarity plus scaled numeric metadata.")

    st.divider()
    if algorithm == "TF-IDF":
        render_recommendation_table(
            "Top 5 TF-IDF Recommendations",
            tfidf_recommendations(courses, selected_title, n=5),
        )
    elif algorithm == "KNN":
        render_recommendation_table(
            "Top 5 KNN Recommendations",
            knn_recommendations(courses, selected_title, n=5),
        )
    else:
        tfidf_col, knn_col = st.columns(2)
        with tfidf_col:
            render_recommendation_table(
                "TF-IDF Recommendations",
                tfidf_recommendations(courses, selected_title, n=5),
            )
        with knn_col:
            render_recommendation_table(
                "KNN Recommendations",
                knn_recommendations(courses, selected_title, n=5),
            )


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
