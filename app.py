from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    "Learner Profiles",
    "About This Demo",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f6fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #667085;
            --line: #d7e0ec;
            --blue: #1d5fd1;
            --teal: #0f766e;
            --green: #047857;
            --amber: #b45309;
            --sidebar: #0b1220;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 95, 209, 0.10), transparent 30rem),
                linear-gradient(180deg, #f7f9fd 0%, #eef3f8 100%);
            color: var(--ink);
        }

        div[data-testid="stDecoration"],
        .stDeployButton {
            display: none;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 3rem;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #0b1220 0%, #111827 58%, #0f1f2f 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
        }

        section[data-testid="stSidebar"] h1 {
            font-size: 1.55rem;
            line-height: 1.15;
            margin-bottom: 1rem;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.10);
            margin: 1.6rem 0 1.3rem 0;
        }

        div[role="radiogroup"] label {
            border-radius: 12px;
            padding: 0.28rem 0.55rem;
            margin-bottom: 0.25rem;
            transition: background 140ms ease, transform 140ms ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(37, 99, 235, 0.22);
            box-shadow: inset 3px 0 0 #60a5fa;
        }

        .app-title {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, #1746b8 0%, #116d7a 58%, #0f766e 100%);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 18px;
            padding: 1.7rem 1.9rem;
            color: #ffffff;
            margin-bottom: 1.25rem;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
        }

        .app-title::after {
            content: "";
            position: absolute;
            inset: auto -8rem -9rem auto;
            width: 22rem;
            height: 22rem;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.11);
            pointer-events: none;
        }

        .app-title h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.12;
            font-weight: 800;
        }

        .app-title p {
            position: relative;
            z-index: 1;
            margin: 0.55rem 0 0 0;
            color: rgba(255, 255, 255, 0.88);
            font-size: 1rem;
            line-height: 1.5;
        }

        .course-card, .profile-card, .mini-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.05rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            margin-bottom: 0.8rem;
        }

        .equal-card {
            min-height: 190px;
            height: 100%;
        }

        .course-card {
            border-left: 4px solid rgba(29, 95, 209, 0.72);
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
            background: #edf6ff;
            color: #1746b8;
            border: 1px solid #c8def8;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .score {
            color: var(--green);
            font-weight: 800;
        }

        .note {
            border-left: 4px solid var(--teal);
            background: #ecfdf5;
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
            color: #064e3b;
            margin: 0.7rem 0 1rem 0;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .home-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.8rem 0;
        }

        .stat-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .stat-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 0.3rem;
        }

        .stat-card:nth-child(1) {
            border-top: 4px solid #1d5fd1;
        }

        .stat-card:nth-child(2) {
            border-top: 4px solid #0f766e;
        }

        .stat-card:nth-child(3) {
            border-top: 4px solid #b45309;
        }

        .home-actions {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .home-action-title {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .flow-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.1rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .flow-card h3 {
            margin: 0 0 0.8rem 0;
            color: var(--ink);
        }

        .flow-row {
            display: flex;
            align-items: stretch;
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .flow-step {
            flex: 1 1 135px;
            min-height: 82px;
            border: 1px solid #c8def8;
            background: #f8fbff;
            border-radius: 12px;
            padding: 0.75rem;
        }

        .flow-step strong {
            display: block;
            color: #1746b8;
            margin-bottom: 0.25rem;
        }

        .flow-step span {
            display: block;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .flow-arrow {
            align-self: center;
            color: var(--blue);
            font-weight: 900;
            font-size: 1.35rem;
        }

        .matrix-demo {
            display: grid;
            grid-template-columns: repeat(5, 18px);
            grid-auto-rows: 18px;
            gap: 4px;
            margin-top: 0.45rem;
        }

        .matrix-demo span {
            border-radius: 4px;
            background: #dbeafe;
            border: 1px solid #bfdbfe;
        }

        .matrix-demo span:nth-child(3n) {
            background: #99f6e4;
            border-color: #5eead4;
        }

        .matrix-demo span:nth-child(4n) {
            background: #fde68a;
            border-color: #fcd34d;
        }

        .score-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }

        .score-box {
            border-radius: 12px;
            padding: 0.85rem;
            background: #f8fbff;
            border: 1px solid #d7e0ec;
        }

        .score-box strong {
            display: block;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .score-box span {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.35;
        }

        .ablation-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.7rem 0 1rem 0;
        }

        .ablation-step {
            border-radius: 12px;
            border: 1px solid #d7e0ec;
            background: #ffffff;
            padding: 0.9rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .ablation-step strong {
            display: block;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .ablation-step span {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        @media (max-width: 900px) {
            .home-grid,
            .home-actions,
            .score-strip,
            .ablation-grid {
                grid-template-columns: 1fr;
            }

            .flow-arrow {
                display: none;
            }
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


def scroll_to_anchor(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
        function scrollToResults() {{
            const anchor = window.parent.document.getElementById("{anchor_id}");
            if (anchor) {{
                anchor.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }}
        setTimeout(scrollToResults, 150);
        setTimeout(scrollToResults, 550);
        </script>
        """,
        height=0,
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


def format_duration(value: int | float | str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return str(value)

    if hours <= 0:
        return ""
    if hours < 1:
        minutes = max(1, round(hours * 60))
        return f"{minutes} min"
    if hours.is_integer():
        return f"{int(hours)} hrs"
    return f"{hours:.1f} hrs"


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
    st.markdown(
        f"""
        <div class="home-grid">
            <div class="stat-card">
                <div class="stat-label">Courses</div>
                <div class="stat-value">{format_number(len(courses))}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Course Subjects</div>
                <div class="stat-value">{format_number(courses["subject"].nunique())}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Difficulty Levels</div>
                <div class="stat-value">{format_number(courses["level"].nunique())}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def ablation_text(courses: pd.DataFrame, feature_set: str) -> pd.Series:
    title = courses["course_title"].fillna("")
    subject = courses["subject"].fillna("")
    level = courses["level"].fillna("")

    if feature_set == "title":
        return title
    if feature_set == "title_subject":
        return title + " " + subject
    return title + " " + subject + " " + level


@st.cache_resource
def build_ablation_recommender(courses: pd.DataFrame, feature_set: str):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(ablation_text(courses, feature_set))
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


def ablation_recommendations(
    courses: pd.DataFrame, selected_title: str, feature_set: str, n: int = 3
) -> pd.DataFrame:
    _, _, similarity_matrix = build_ablation_recommender(courses, feature_set)
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
        formatted_duration = format_duration(duration)
        if formatted_duration:
            duration_html = f" · {escape(formatted_duration)}"
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


def course_picker(
    courses: pd.DataFrame,
    key_prefix: str,
) -> tuple[str, str]:
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

    subjects = ", ".join(sorted(courses["subject"].dropna().unique()))
    levels = ", ".join(sorted(courses["level"].dropna().unique()))
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="home-action-title">Dataset Coverage</div>
            <div class="meta">These subjects and levels come from the processed Udemy course metadata.</div>
            <div class="meta"><strong>Subjects:</strong> {escape(subjects)}</div>
            <div class="meta"><strong>Levels:</strong> {escape(levels)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-actions">
            <div class="mini-card">
                <div class="home-action-title">Start With a Course</div>
                <div class="meta">Search for a real Udemy course title, choose TF-IDF, KNN, or Compare Both, and review ranked course cards.</div>
            </div>
            <div class="mini-card">
                <div class="home-action-title">Try a Learner Profile</div>
                <div class="meta">Select a synthetic learner persona to demonstrate how metadata-based personalization could appear in the interface.</div>
            </div>
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
            key="course_algorithm",
        )
        count = st.slider(
            "Number of recommendations",
            3,
            10,
            5,
            key="course_recommendation_count",
        )
        generate = st.button(
            "Generate recommendations",
            type="primary",
            use_container_width=True,
        )
        st.caption("Change the course or algorithm, then click Generate to update results.")

    if generate:
        st.session_state["course_result_request"] = {
            "query": query,
            "algorithm": algorithm,
            "count": count,
        }
        st.session_state["course_results_just_generated"] = True

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

    request = st.session_state.get("course_result_request")
    if request is None:
        st.info("Choose a course and click Generate recommendations to view results.")
        return

    current_signature = (matched_title, algorithm, count)
    request_query = str(request["query"])
    request_algorithm = str(request["algorithm"])
    request_count = int(request["count"])

    try:
        _, request_title, _ = resolve_course_selection(courses, request_query)
    except ValueError as error:
        st.warning(str(error))
        return

    request_signature = (request_title, request_algorithm, request_count)
    if current_signature != request_signature:
        st.info("Options changed. Click Generate recommendations to update the results below.")

    st.divider()
    st.markdown('<div id="recommendation-results"></div>', unsafe_allow_html=True)
    st.success(f"Results loaded for '{request_title}' using {request_algorithm}.")
    should_scroll = bool(st.session_state.pop("course_results_just_generated", False))
    if should_scroll:
        scroll_to_anchor("recommendation-results")

    if request_algorithm == "TF-IDF + Cosine Similarity":
        st.subheader("Recommended Courses")
        render_recommendation_cards(
            tfidf_recommendations(courses, request_title, n=request_count),
            label="Similarity",
        )
    elif request_algorithm == "KNN Recommender":
        st.subheader("Recommended Courses")
        render_recommendation_cards(
            knn_recommendations(courses, request_title, n=request_count),
            label="KNN score",
        )
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("TF-IDF + Cosine Similarity")
            render_recommendation_cards(
                tfidf_recommendations(courses, request_title, n=request_count),
                label="Similarity",
            )
        with right:
            st.subheader("KNN Recommender")
            render_recommendation_cards(
                knn_recommendations(courses, request_title, n=request_count),
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
    generate = st.button(
        "Generate profile recommendations",
        type="primary",
        use_container_width=True,
    )
    if generate:
        st.session_state["profile_result_request"] = {
            "persona_name": selected_name,
            "count": count,
        }
        st.session_state["profile_results_just_generated"] = True

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(
            f"""
            <div class="profile-card equal-card">
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
            <div class="mini-card equal-card">
                <h3>How the demo matches courses</h3>
                <div class="meta">
                    The app compares profile interests, preferred subject, and preferred level with course metadata.
                    No real user histories, ratings, or interactions are used.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    request = st.session_state.get("profile_result_request")
    if request is None:
        st.info("Choose a learner profile and click Generate profile recommendations to view results.")
        return

    request_name = str(request["persona_name"])
    request_count = int(request["count"])
    if selected_name != request_name or count != request_count:
        st.info("Profile options changed. Click Generate profile recommendations to update the results below.")

    request_persona = personas[personas["persona_name"] == request_name].iloc[0]
    st.divider()
    st.markdown('<div id="profile-results"></div>', unsafe_allow_html=True)
    st.subheader("Recommended Courses")
    st.success(f"Results loaded for learner profile: {request_name}.")
    if st.session_state.pop("profile_results_just_generated", False):
        scroll_to_anchor("profile-results")
    render_recommendation_cards(
        persona_recommendations(courses, request_persona, n=request_count),
        label="Profile score",
    )


def about_page(courses: pd.DataFrame) -> None:
    render_title(
        "How Recommendations Are Generated",
        "A short explanation of how the app turns course metadata into recommendation results.",
    )
    st.markdown(
        f"""
        <div class="flow-card">
            <h3>Data Used by the App</h3>
            <div class="meta">
                Processed Udemy metadata with {format_number(len(courses))} courses. The app uses title,
                subject, level, subscribers, reviews, price, duration, and a combined text feature.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <h3>TF-IDF + Cosine Similarity</h3>
            <div class="flow-row">
                <div class="flow-step">
                    <strong>Course Text</strong>
                    <span>title + subject + level combined into one text field</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>TF-IDF Matrix</strong>
                    <span>each course becomes a weighted word vector</span>
                    <div class="matrix-demo">
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Cosine Similarity</strong>
                    <span>compares the selected course vector with all other course vectors</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Top Courses</strong>
                    <span>highest similarity scores become recommendations</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <h3>KNN Recommender</h3>
            <div class="flow-row">
                <div class="flow-step">
                    <strong>Text Features</strong>
                    <span>TF-IDF vector from combined course text</span>
                </div>
                <div class="flow-arrow">+</div>
                <div class="flow-step">
                    <strong>Numeric Features</strong>
                    <span>subscribers, reviews, price, and duration are scaled</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Combined Matrix</strong>
                    <span>text and metadata form one course representation</span>
                    <div class="matrix-demo">
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Nearest Neighbors</strong>
                    <span>KNN returns courses closest to the selected course</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <h3>Learner Profile Recommendations</h3>
            <div class="flow-row">
                <div class="flow-step">
                    <strong>Profile</strong>
                    <span>interests, preferred subject, preferred level</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Course Metadata</strong>
                    <span>title, subject, level, combined text</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Match Score</strong>
                    <span>interest terms + subject match + level match</span>
                </div>
                <div class="flow-arrow">→</div>
                <div class="flow-step">
                    <strong>Recommended Courses</strong>
                    <span>highest profile scores are shown first</span>
                </div>
            </div>
            <div class="meta" style="margin-top: 0.75rem;">
                These learner profiles are synthetic and are used only for demonstration. This is not collaborative filtering.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <h3>How to Read the Scores</h3>
            <div class="score-strip">
                <div class="score-box">
                    <strong>TF-IDF Similarity</strong>
                    <span>Higher means stronger text/topic similarity between course descriptions.</span>
                </div>
                <div class="score-box">
                    <strong>KNN Score</strong>
                    <span>Shown as 1 − cosine distance. Higher means closer in text + metadata space.</span>
                </div>
                <div class="score-box">
                    <strong>Profile Score</strong>
                    <span>Higher means more matches with the synthetic learner's interests, subject, and level.</span>
                </div>
            </div>
            <div class="meta" style="margin-top: 0.8rem;">
                Collaborative filtering is not used because the dataset does not include real user behavior such as ratings,
                clicks, enrollments, or watch history.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-card">
            <h3>Interactive Feature Ablation Demo</h3>
            <div class="meta">
                This demo repeats the ablation idea from the report for any selected course. It temporarily changes
                the TF-IDF text representation so you can see how recommendations and similarity scores shift as more
                course metadata is included. The final recommender uses title + subject + level.
            </div>
            <div class="ablation-grid">
                <div class="ablation-step">
                    <strong>Title Only</strong>
                    <span>Uses the course title words as the only text signal.</span>
                </div>
                <div class="ablation-step">
                    <strong>Title + Subject</strong>
                    <span>Adds the Udemy subject category to reinforce topic area.</span>
                </div>
                <div class="ablation-step">
                    <strong>Title + Subject + Level</strong>
                    <span>Adds difficulty context and matches the final combined_features text.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ablation_query, _ = course_picker(courses, "ablation")
    ablation_count = st.slider(
        "Recommendations per feature set",
        min_value=3,
        max_value=5,
        value=3,
        key="ablation_count",
    )

    current_index, current_matched_title, current_partial_match = resolve_course_selection(
        courses, ablation_query
    )

    if st.button("Generate ablation results", key="run_ablation", type="primary"):
        selected_index, matched_title, partial_match = resolve_course_selection(
            courses, ablation_query
        )
        st.session_state["ablation_request"] = {
            "query": ablation_query,
            "title": matched_title,
            "count": ablation_count,
            "partial_match": partial_match,
        }

    ablation_request = st.session_state.get("ablation_request")
    if ablation_request is None:
        st.info("Select a course and run the ablation demo to compare feature choices.")
        return

    matched_title = str(ablation_request["title"])
    count = int(ablation_request["count"])
    current_signature = (current_matched_title, ablation_count)
    request_signature = (matched_title, count)
    if current_signature != request_signature:
        st.info(
            "Course or recommendation count changed. Click Generate ablation results to update this comparison."
        )

    if ablation_request.get("partial_match"):
        st.caption(f"Matched closest title: {matched_title}")
    st.success(
        f"Ablation results loaded for '{matched_title}'. The final recommender uses title + subject + level."
    )

    feature_sets = [
        ("Title only", "title"),
        ("Title + subject", "title_subject"),
        ("Title + subject + level", "title_subject_level"),
    ]
    columns = st.columns(3)
    for column, (heading, feature_set) in zip(columns, feature_sets):
        with column:
            st.markdown(f"### {heading}")
            render_recommendation_cards(
                ablation_recommendations(courses, matched_title, feature_set, n=count),
                label="Similarity",
            )


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
    elif page == "Learner Profiles":
        learner_profile_page(courses)
    elif page == "About This Demo":
        about_page(courses)


if __name__ == "__main__":
    main()
