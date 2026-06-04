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
RECOMMENDATION_EXAMPLES_PATH = (
    ROOT / "results" / "recommendation_examples" / "tfidf_knn_evaluation_examples.csv"
)
EVALUATION_DIR = ROOT / "results" / "evaluation"
GRAPHS_DIR = ROOT / "results" / "graphs"
SYNTHETIC_USERS_PATH = ROOT / "data" / "synthetic_users.csv"


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

        .section-kicker {
            color: var(--blue);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .soft-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            text-align: center;
            font-size: 0.9rem;
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


def render_section_intro(kicker: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="section-kicker">{kicker}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_courses() -> pd.DataFrame:
    if not PROCESSED_COURSES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_COURSES_PATH)


@st.cache_data
def load_recommendation_examples() -> pd.DataFrame:
    if not RECOMMENDATION_EXAMPLES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RECOMMENDATION_EXAMPLES_PATH)


@st.cache_data
def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_synthetic_users() -> pd.DataFrame:
    return load_csv_if_exists(SYNTHETIC_USERS_PATH)


@st.cache_data
def load_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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


def persona_recommendations(courses: pd.DataFrame, persona: pd.Series, n: int = 5) -> pd.DataFrame:
    interests = str(persona["interests"]).lower()
    interest_terms = [
        token.strip()
        for phrase in interests.split(";")
        for token in phrase.split()
        if len(token.strip()) > 2
    ]
    preferred_subject = str(persona["preferred_subject"]).lower()
    preferred_level = str(persona["preferred_level"]).lower()

    scored = courses.copy()
    text = (
        scored["course_title"].fillna("")
        + " "
        + scored["subject"].fillna("")
        + " "
        + scored["level"].fillna("")
        + " "
        + scored["combined_features"].fillna("")
    ).str.lower()

    scored["interest_matches"] = text.apply(
        lambda value: sum(1 for term in interest_terms if term in value)
    )
    scored["subject_match"] = (
        scored["subject"].fillna("").str.lower() == preferred_subject
    ).astype(int)
    scored["level_match"] = (
        scored["level"].fillna("").str.lower() == preferred_level
    ).astype(int)
    scored["persona_score"] = (
        scored["interest_matches"]
        + scored["subject_match"] * 2
        + scored["level_match"]
    )

    result = scored.sort_values(
        ["persona_score", "interest_matches", "num_subscribers"],
        ascending=[False, False, False],
    ).head(n)
    result = result[
        [
            "course_title",
            "subject",
            "level",
            "num_subscribers",
            "num_reviews",
            "price",
            "persona_score",
        ]
    ].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def render_recommendation_table(title: str, recommendations: pd.DataFrame) -> None:
    st.subheader(title)
    if recommendations.empty:
        st.info("No recommendations available.")
        return
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.4f"),
            "distance": st.column_config.NumberColumn("distance", format="%.4f"),
            "num_subscribers": st.column_config.NumberColumn("num_subscribers", format="%d"),
            "num_reviews": st.column_config.NumberColumn("num_reviews", format="%d"),
            "price": st.column_config.NumberColumn("price", format="%d"),
            "persona_score": st.column_config.NumberColumn("persona_score", format="%d"),
        },
    )


def project_overview_section() -> None:
    st.header("Project Overview")
    render_section_intro(
        "Final Project Dashboard",
        "This dashboard brings together the processed course dataset, real recommender outputs, evaluation files, exported graphs, and synthetic learner personas.",
    )
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
    render_section_intro(
        "Live Content-Based Recommendation",
        "Select any processed course and generate Top-5 recommendations using TF-IDF, KNN, or both models side by side.",
    )
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
    render_section_intro(
        "Saved Evaluation Examples",
        "This section uses the generated TF-IDF and KNN recommendation examples that were saved for the evaluation notebook.",
    )
    examples = load_recommendation_examples()
    if examples.empty:
        st.warning(f"Recommendation examples not found at `{RECOMMENDATION_EXAMPLES_PATH}`.")
        return

    st.markdown(
        "This section uses the saved comparison outputs generated for the evaluation notebook."
    )
    category = st.selectbox(
        "Evaluation example",
        examples["category"].drop_duplicates().tolist(),
    )
    subset = examples[examples["category"] == category].copy()
    input_course = subset["input_course"].iloc[0]
    st.markdown(f"**Input course:** `{input_course}`")

    tfidf_rows = subset[subset["model"] == "TF-IDF"].drop(columns=["category", "input_course"])
    knn_rows = subset[subset["model"] == "KNN"].drop(columns=["category", "input_course"])

    left, right = st.columns(2)
    with left:
        st.subheader("TF-IDF Recommendations")
        st.dataframe(tfidf_rows, use_container_width=True, hide_index=True)
    with right:
        st.subheader("KNN Recommendations")
        st.dataframe(knn_rows, use_container_width=True, hide_index=True)

    tfidf_titles = set(tfidf_rows["course_title"])
    knn_titles = set(knn_rows["course_title"])
    common_titles = sorted(tfidf_titles.intersection(knn_titles))
    overlap = len(common_titles) / 5 * 100

    st.subheader("Overlap Discussion")
    st.metric("Common Recommendations", f"{len(common_titles)} / 5", f"{overlap:.0f}% overlap")
    if common_titles:
        st.write("**Shared recommendations:**")
        for title in common_titles:
            st.write(f"- {title}")
    st.write(
        "Higher overlap means both algorithms agree on similar courses. Lower overlap can indicate that KNN is introducing more varied alternatives because it also uses numeric metadata."
    )


def evaluation_results_section() -> None:
    st.header("Evaluation Results")
    render_section_intro(
        "Measured Project Outputs",
        "These tables come from the completed evaluation exports and summarize overlap, runtime, and ablation findings.",
    )
    overlap = load_csv_if_exists(EVALUATION_DIR / "recommendation_overlap.csv")
    runtime = load_csv_if_exists(EVALUATION_DIR / "runtime_comparison.csv")
    ablation = load_csv_if_exists(EVALUATION_DIR / "ablation_results.csv")
    summary = load_text_if_exists(EVALUATION_DIR / "evaluation_summary.md")

    st.markdown("Evaluation is based on the completed notebook results and exported CSV files.")

    tab_overlap, tab_runtime, tab_ablation, tab_summary = st.tabs(
        ["Overlap", "Runtime", "Ablation", "Summary"]
    )
    with tab_overlap:
        st.subheader("Recommendation Overlap")
        if overlap.empty:
            st.warning("Overlap results are not available.")
        else:
            st.dataframe(overlap, use_container_width=True, hide_index=True)
    with tab_runtime:
        st.subheader("Runtime Comparison")
        if runtime.empty:
            st.warning("Runtime results are not available.")
        else:
            st.dataframe(runtime, use_container_width=True, hide_index=True)
    with tab_ablation:
        st.subheader("Ablation Study")
        if ablation.empty:
            st.warning("Ablation results are not available.")
        else:
            st.dataframe(ablation, use_container_width=True, hide_index=True)
    with tab_summary:
        st.subheader("Evaluation Summary")
        if summary:
            st.markdown(summary)
        else:
            st.warning("Evaluation summary is not available.")


def synthetic_profiles_section() -> None:
    st.header("Synthetic Learner Profiles")
    render_section_intro(
        "Persona Demonstration",
        "Synthetic personas demonstrate how a future interface could personalize a course discovery experience without using real user histories.",
    )
    st.markdown(
        """
        <div class="note">
            Synthetic learner profiles are for demonstration only. This is not collaborative filtering.
        </div>
        """,
        unsafe_allow_html=True,
    )
    personas = load_synthetic_users()
    courses = load_courses()
    if personas.empty:
        st.warning(f"Synthetic persona file not found at `{SYNTHETIC_USERS_PATH}`.")
        return
    if courses.empty:
        st.warning(f"Processed dataset not found at `{PROCESSED_COURSES_PATH}`.")
        return

    selected_persona_name = st.selectbox(
        "Select a learner persona",
        personas["persona_name"].tolist(),
    )
    persona = personas[personas["persona_name"] == selected_persona_name].iloc[0]

    left, right = st.columns([0.9, 1.2])
    with left:
        st.subheader("Persona Details")
        st.markdown(
            f"""
            <div class="card">
                <h3>{persona["persona_name"]}</h3>
                <span class="badge">{persona["preferred_subject"]}</span>
                <span class="badge">{persona["preferred_level"]}</span>
                <p><strong>Interests:</strong> {persona["interests"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Simple Matching Method")
        st.write(
            "The demo ranks courses by matching persona interests with course titles, subjects, levels, and combined features."
        )
        st.write(
            "These are synthetic learner profiles for demonstration only. This is not collaborative filtering."
        )

    st.divider()
    render_recommendation_table(
        "Persona-Based Content Recommendations",
        persona_recommendations(courses, persona, n=5),
    )


def graphs_section() -> None:
    st.header("Graphs and Results")
    render_section_intro(
        "Presentation Assets",
        "This section displays exported graph files and their summary notes for direct use in the final presentation.",
    )
    if not GRAPHS_DIR.exists():
        st.warning(f"Graphs directory not found at `{GRAPHS_DIR}`.")
        return

    graph_files = sorted(GRAPHS_DIR.glob("*.png"))
    if not graph_files:
        st.info("No graph images are currently available.")
    else:
        st.markdown("These visuals come from the project notebooks and evaluation results.")
        for index in range(0, len(graph_files), 2):
            cols = st.columns(2)
            for col, graph_path in zip(cols, graph_files[index : index + 2]):
                with col:
                    title = graph_path.stem.replace("_", " ").title()
                    st.subheader(title)
                    st.image(str(graph_path), use_container_width=True)

    graph_summary = load_text_if_exists(GRAPHS_DIR / "graph_summary.md")
    if graph_summary:
        st.divider()
        st.subheader("Graph Summary")
        st.markdown(graph_summary)


def limitations_section() -> None:
    st.header("Limitations")
    render_section_intro(
        "Responsible Interpretation",
        "The project uses available metadata and synthetic personas only, so the results should be interpreted as content-based recommendation behavior rather than user-preference learning.",
    )
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

    st.markdown(
        """
        <div class="footer">
            Online Course Recommender System | Content-based TF-IDF and KNN project dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
