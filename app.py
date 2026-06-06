from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Online Course Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
PROCESSED_COURSES_PATH = ROOT / "data" / "processed" / "processed_udemy_courses.csv"
SYNTHETIC_USERS_PATH = ROOT / "data" / "synthetic_users.csv"
GRAPHS_DIR = ROOT / "results" / "graphs"


PAGES = [
    "Home",
    "Course Recommender",
    "Learner Profile Recommendations",
    "Compare Algorithms",
    "About This Demo",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #667085;
            --line: #d9e2ef;
            --blue: #2563eb;
            --green: #047857;
            --amber: #b45309;
        }

        .stApp {
            background: #f6f8fb;
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.3rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #101828;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .app-title {
            background: linear-gradient(135deg, #1f4ed8 0%, #0f766e 100%);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            color: #ffffff;
            margin-bottom: 1.1rem;
            box-shadow: 0 16px 42px rgba(15, 23, 42, 0.14);
        }

        .app-title h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.12;
            font-weight: 800;
        }

        .app-title p {
            margin: 0.55rem 0 0 0;
            color: rgba(255, 255, 255, 0.88);
            font-size: 1rem;
            line-height: 1.5;
        }

        .course-card, .profile-card, .mini-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }

        .course-card h3, .profile-card h3, .mini-card h3 {
            color: var(--ink);
            font-size: 1.05rem;
            line-height: 1.25;
            margin: 0 0 0.55rem 0;
        }

        .meta {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            margin: 0.1rem 0.2rem 0.2rem 0;
            background: #eef4ff;
            color: #1d4ed8;
            border: 1px solid #c7d7fe;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .score {
            color: var(--green);
            font-weight: 800;
        }

        .note {
            border-left: 4px solid var(--blue);
            background: #eff6ff;
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
            color: #1e3a8a;
            margin: 0.7rem 0 1rem 0;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-title">
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
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
def load_synthetic_users() -> pd.DataFrame:
    if not SYNTHETIC_USERS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SYNTHETIC_USERS_PATH)


def format_number(value: int | float | str) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def format_price(value: int | float | str) -> str:
    try:
        return f"${int(float(value))}"
    except (TypeError, ValueError):
        return str(value)


def required_columns_available(courses: pd.DataFrame) -> bool:
    required = {
        "course_title",
        "subject",
        "level",
        "num_subscribers",
        "num_reviews",
        "price",
        "combined_features",
    }
    return required.issubset(courses.columns)


def render_dataset_stats(courses: pd.DataFrame) -> None:
    cols = st.columns(3)
    cols[0].metric("Courses", format_number(len(courses)))
    cols[1].metric("Subjects", format_number(courses["subject"].nunique()))
    cols[2].metric("Levels", format_number(courses["level"].nunique()))


def resolve_course_selection(courses: pd.DataFrame, title: str) -> tuple[int, str, bool]:
    query = str(title).strip().lower()
    if not query:
        raise ValueError("Please search for or select a course title.")

    normalized_titles = courses["course_title"].fillna("").str.lower()
    exact_matches = courses.index[normalized_titles == query].tolist()
    if exact_matches:
        index = exact_matches[0]
        return index, str(courses.loc[index, "course_title"]), False

    partial_matches = courses.index[
        normalized_titles.str.contains(query, regex=False, na=False)
    ].tolist()
    if partial_matches:
        index = partial_matches[0]
        return index, str(courses.loc[index, "course_title"]), True

    raise ValueError(f"No course title matched: {title}")


def get_course_index(courses: pd.DataFrame, title: str) -> int:
    index, _, _ = resolve_course_selection(courses, title)
    return index


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
        feature_matrix = hstack([text_matrix, numeric_matrix * 0.2]).tocsr()
    else:
        feature_matrix = text_matrix.tocsr()

    model = NearestNeighbors(n_neighbors=51, metric="cosine", algorithm="brute")
    model.fit(feature_matrix)
    return model, feature_matrix


def recommendation_columns(courses: pd.DataFrame) -> list[str]:
    columns = [
        "course_title",
        "subject",
        "level",
        "num_subscribers",
        "num_reviews",
        "price",
    ]
    if "content_duration" in courses.columns:
        columns.append("content_duration")
    return columns


def tfidf_recommendations(
    courses: pd.DataFrame, selected_title: str, n: int = 5
) -> pd.DataFrame:
    _, _, similarity_matrix = build_tfidf_recommender(courses)
    selected_index = get_course_index(courses, selected_title)
    scores = list(enumerate(similarity_matrix[selected_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != selected_index][:n]

    result = courses.loc[[index for index, _ in scores], recommendation_columns(courses)].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["score"] = [round(float(score), 4) for _, score in scores]
    return result.reset_index(drop=True)


def knn_recommendations(
    courses: pd.DataFrame, selected_title: str, n: int = 5
) -> pd.DataFrame:
    model, feature_matrix = build_knn_recommender(courses)
    selected_index = get_course_index(courses, selected_title)
    distances, indices = model.kneighbors(
        feature_matrix[selected_index],
        n_neighbors=min(n + 1, len(courses)),
    )

    rows = []
    for distance, index in zip(distances[0], indices[0]):
        if index == selected_index:
            continue
        row = courses.loc[index, recommendation_columns(courses)].to_dict()
        row["distance"] = round(float(distance), 4)
        row["score"] = round(1 - float(distance), 4)
        rows.append(row)
        if len(rows) == n:
            break

    result = pd.DataFrame(rows)
    if not result.empty:
        result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


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
    scored["score"] = (
        scored["interest_matches"]
        + scored["subject_match"] * 2
        + scored["level_match"]
    )

    result = scored.sort_values(
        ["score", "interest_matches", "num_subscribers"],
        ascending=[False, False, False],
    ).head(n)
    result = result[recommendation_columns(courses) + ["score"]].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def render_course_card(
    course: pd.Series | dict,
    *,
    rank: int | None = None,
    score: float | None = None,
    label: str = "Similarity",
) -> None:
    title = escape(str(course.get("course_title", "Untitled course")))
    subject = escape(str(course.get("subject", "Unknown subject")))
    level = escape(str(course.get("level", "Unknown level")))
    subscribers = format_number(course.get("num_subscribers", 0))
    reviews = format_number(course.get("num_reviews", 0))
    price = format_price(course.get("price", 0))
    duration = course.get("content_duration")
    duration_html = ""
    if duration is not None:
        duration_html = f" · {escape(str(duration))} hrs"
    rank_html = f"<span class='badge'>#{rank}</span>" if rank is not None else ""
    score_html = ""
    if score is not None:
        score_html = f"<div class='score'>{escape(label)}: {score:.4f}</div>"

    st.markdown(
        f"""
        <div class="course-card">
            <h3>{rank_html} {title}</h3>
            <span class="badge">{subject}</span>
            <span class="badge">{level}</span>
            <div class="meta">
                {subscribers} subscribers · {reviews} reviews · {price}{duration_html}
            </div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_cards(
    recommendations: pd.DataFrame, *, label: str = "Similarity"
) -> None:
    if recommendations.empty:
        st.info("No recommendations available.")
        return

    for _, row in recommendations.iterrows():
        render_course_card(
            row.to_dict(),
            rank=int(row["rank"]),
            score=float(row["score"]) if "score" in row else None,
            label=label,
        )


def course_picker(courses: pd.DataFrame, key_prefix: str) -> tuple[str, str]:
    search = st.text_input(
        "Search course title",
        placeholder="Try python, investment banking, web development...",
        key=f"{key_prefix}_search",
    )
    titles = courses["course_title"].dropna().sort_values().tolist()
    if search.strip():
        filtered = [
            title for title in titles if search.strip().lower() in title.lower()
        ]
        if not filtered:
            st.warning("No visible title matches the search yet. You can still use the typed query.")
            filtered = titles[:100]
    else:
        filtered = titles[:250]

    selected_title = st.selectbox(
        "Select a course",
        filtered,
        key=f"{key_prefix}_select",
    )
    query = search.strip() or selected_title
    return query, selected_title


def home_page(courses: pd.DataFrame) -> None:
    render_title(
        "Find Similar Udemy Courses",
        "Select a course or learner profile to receive recommended Udemy courses.",
    )
    st.write(
        "Use the recommender to explore course-based matches or try a synthetic learner profile for a personalized demo."
    )
    render_dataset_stats(courses)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="mini-card">
                <h3>Course Recommender</h3>
                <div class="meta">Search for a real course and receive similar courses using TF-IDF, KNN, or both.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="mini-card">
                <h3>Learner Profiles</h3>
                <div class="meta">Choose a synthetic learner profile and see metadata-based course suggestions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def course_recommender_page(courses: pd.DataFrame) -> None:
    render_title(
        "Course Recommender",
        "Search for a course and generate similar Udemy course recommendations.",
    )

    controls, selected_panel = st.columns([0.95, 1.05])
    with controls:
        query, _ = course_picker(courses, "course")
        algorithm = st.radio(
            "Recommendation algorithm",
            ["TF-IDF + Cosine Similarity", "KNN Recommender", "Compare Both"],
        )
        count = st.slider("Number of recommendations", 3, 10, 5)

    try:
        selected_index, matched_title, partial_match = resolve_course_selection(courses, query)
    except ValueError as error:
        st.warning(str(error))
        return

    with selected_panel:
        st.subheader("Selected Course")
        if partial_match:
            st.caption(f"Matched closest title: {matched_title}")
        render_course_card(courses.loc[selected_index])

    st.divider()
    if algorithm == "TF-IDF + Cosine Similarity":
        st.subheader("Recommended Courses")
        render_recommendation_cards(
            tfidf_recommendations(courses, matched_title, n=count),
            label="Similarity",
        )
    elif algorithm == "KNN Recommender":
        st.subheader("Recommended Courses")
        render_recommendation_cards(
            knn_recommendations(courses, matched_title, n=count),
            label="KNN score",
        )
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("TF-IDF + Cosine Similarity")
            render_recommendation_cards(
                tfidf_recommendations(courses, matched_title, n=count),
                label="Similarity",
            )
        with right:
            st.subheader("KNN Recommender")
            render_recommendation_cards(
                knn_recommendations(courses, matched_title, n=count),
                label="KNN score",
            )


def learner_profile_page(courses: pd.DataFrame) -> None:
    render_title(
        "Learner Profile Recommendations",
        "Choose a synthetic learner profile and receive course suggestions based on interests and metadata.",
    )
    st.markdown(
        """
        <div class="note">
            These are synthetic learner profiles for demonstration only. This is not collaborative filtering.
        </div>
        """,
        unsafe_allow_html=True,
    )

    personas = load_synthetic_users()
    if personas.empty:
        st.warning(f"Synthetic learner profile file not found at `{SYNTHETIC_USERS_PATH}`.")
        return

    selected_name = st.selectbox("Learner profile", personas["persona_name"].tolist())
    persona = personas[personas["persona_name"] == selected_name].iloc[0]
    count = st.slider("Number of recommendations", 3, 10, 5, key="persona_count")

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(
            f"""
            <div class="profile-card">
                <h3>{escape(str(persona["persona_name"]))}</h3>
                <span class="badge">{escape(str(persona["preferred_subject"]))}</span>
                <span class="badge">{escape(str(persona["preferred_level"]))}</span>
                <div class="meta"><strong>Interests:</strong> {escape(str(persona["interests"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="mini-card">
                <h3>How the demo matches courses</h3>
                <div class="meta">
                    The app compares profile interests, preferred subject, and preferred level with course metadata.
                    No real user histories, ratings, or interactions are used.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Recommended Courses")
    render_recommendation_cards(
        persona_recommendations(courses, persona, n=count),
        label="Profile score",
    )


def compare_algorithms_page(courses: pd.DataFrame) -> None:
    render_title(
        "Compare Algorithms",
        "Select one course and compare the two content-based recommendation methods side by side.",
    )
    query, _ = course_picker(courses, "compare")
    count = st.slider("Number of recommendations", 3, 10, 5, key="compare_count")

    try:
        _, matched_title, partial_match = resolve_course_selection(courses, query)
    except ValueError as error:
        st.warning(str(error))
        return

    if partial_match:
        st.caption(f"Matched closest title: {matched_title}")

    left, right = st.columns(2)
    with left:
        st.subheader("TF-IDF + Cosine Similarity")
        st.caption("TF-IDF focuses mainly on textual similarity.")
        render_recommendation_cards(
            tfidf_recommendations(courses, matched_title, n=count),
            label="Similarity",
        )
    with right:
        st.subheader("KNN Recommender")
        st.caption("KNN combines text features with numeric course metadata.")
        render_recommendation_cards(
            knn_recommendations(courses, matched_title, n=count),
            label="KNN score",
        )


def about_page(courses: pd.DataFrame) -> None:
    render_title(
        "About This Demo",
        "A short overview of the data and methods behind the recommender interface.",
    )
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            """
            <div class="mini-card">
                <h3>Dataset</h3>
                <div class="meta">Udemy course metadata including titles, subjects, levels, subscribers, reviews, prices, and duration.</div>
            </div>
            <div class="mini-card">
                <h3>Methods</h3>
                <div class="meta">TF-IDF + cosine similarity and KNN content-based recommendation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="mini-card">
                <h3>Limitation</h3>
                <div class="meta">The dataset has no real user interaction data, so collaborative filtering is not used.</div>
            </div>
            <div class="mini-card">
                <h3>Synthetic Learners</h3>
                <div class="meta">Synthetic profiles are only for demonstrating possible personalization in the interface.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_files = [
        GRAPHS_DIR / "number_of_courses_per_subject_category.png",
        GRAPHS_DIR / "distribution_of_course_levels.png",
        GRAPHS_DIR / "recommendation_overlap_comparison.png",
    ]
    available = [path for path in chart_files if path.exists()]
    if available:
        st.subheader("Optional Project Charts")
        cols = st.columns(len(available))
        for col, path in zip(cols, available):
            with col:
                st.image(str(path), use_container_width=True)


def main() -> None:
    inject_styles()
    courses = load_courses()

    with st.sidebar:
        st.title("Course Recommender")
        page = st.radio("Navigation", PAGES)
        st.divider()
        if not courses.empty and required_columns_available(courses):
            st.caption(f"{format_number(len(courses))} courses loaded")
        st.caption("Content-based TF-IDF and KNN demo")

    if courses.empty:
        render_title("Course Recommender", "The processed course dataset could not be loaded.")
        st.warning(f"Processed dataset not found at `{PROCESSED_COURSES_PATH}`.")
        return
    if not required_columns_available(courses):
        render_title("Course Recommender", "The processed course dataset is missing required columns.")
        st.error("Please check the processed dataset columns before running the demo.")
        st.dataframe(pd.DataFrame({"available_columns": courses.columns}))
        return

    if page == "Home":
        home_page(courses)
    elif page == "Course Recommender":
        course_recommender_page(courses)
    elif page == "Learner Profile Recommendations":
        learner_profile_page(courses)
    elif page == "Compare Algorithms":
        compare_algorithms_page(courses)
    elif page == "About This Demo":
        about_page(courses)


if __name__ == "__main__":
    main()
