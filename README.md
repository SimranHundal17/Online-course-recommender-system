# Online Course Recommender System

A content-based recommender system for Udemy-style online courses. The project compares a TF-IDF + cosine similarity recommender with a KNN-based recommender, evaluates their recommendation behavior, and includes a Streamlit dashboard prototype for course-based and synthetic learner-persona demonstrations.

## Project Overview

This project recommends online courses using course metadata rather than real user interaction history. The dataset includes course titles, subjects, levels, prices, subscribers, reviews, and course duration.

The main recommendation approaches are:

- **TF-IDF + Cosine Similarity:** recommends courses with similar textual features.
- **KNN Recommender:** combines TF-IDF text features with scaled numeric metadata.

The project does **not** use collaborative filtering because the dataset does not contain real user ratings, clicks, enrollments, or watch history.

## Repository Structure

```text
Online-course-recommender-system/
|-- app.py
|-- data/
|   |-- raw/
|   |   `-- udemy_courses.csv
|   |-- processed/
|   |   `-- processed_udemy_courses.csv
|   `-- synthetic_users.csv
|-- notebooks/
|   |-- 01_data_understanding.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_tfidf_recommender.ipynb
|   |-- 04_knn_recommender.ipynb
|   `-- 05_evaluation.ipynb
|-- results/
|   |-- evaluation/
|   |-- graphs/
|   `-- recommendation_examples/
|-- requirements.txt
`-- README.md
```

## Dataset

The raw dataset is stored at:

```text
data/raw/udemy_courses.csv
```

The processed dataset is stored at:

```text
data/processed/processed_udemy_courses.csv
```

The processed dataset contains 3,672 courses and these key fields:

- `course_id`
- `course_title`
- `subject`
- `level`
- `num_subscribers`
- `num_reviews`
- `price`
- `content_duration`
- `combined_features`

The `combined_features` column combines course title, subject, and level. It is the main text representation used by the recommender notebooks.

## Notebooks

### 1. Data Understanding

`notebooks/01_data_understanding.ipynb`

Explores the raw Udemy dataset, including dataset shape, missing values, duplicate rows, subject distribution, subscriber distribution, and course level distribution.

### 2. Preprocessing

`notebooks/02_preprocessing.ipynb`

Creates the cleaned recommendation dataset by selecting useful recommendation features, cleaning text columns, handling missing values, removing duplicate courses, creating `combined_features`, and saving `processed_udemy_courses.csv`.

### 3. TF-IDF Recommender

`notebooks/03_tfidf_recommender.ipynb`

Builds a content-based recommender using TF-IDF vectorization, cosine similarity, and `recommend_courses(course_title, n=5)`. This model acts as the baseline recommender because it is simple, interpretable, and fast.

### 4. KNN Recommender

`notebooks/04_knn_recommender.ipynb`

Builds a KNN-based content recommender using TF-IDF text vectors, scaled numeric course metadata, cosine distance, and nearest-neighbor search.

The KNN similarity score is displayed as:

```text
similarity = 1 - distance
```

### 5. Evaluation

`notebooks/05_evaluation.ipynb`

Compares TF-IDF and KNN using recommendation relevance, recommendation overlap, runtime comparison, ablation study, and comparative discussion.

Traditional supervised metrics such as Accuracy, Precision, Recall, and F1 Score are not used because the dataset does not contain ground-truth user interaction labels.

## Evaluation Results

Evaluation exports are stored in:

```text
results/evaluation/
```

Key result files:

- `recommendation_overlap.csv`
- `runtime_comparison.csv`
- `ablation_results.csv`
- `evaluation_summary.md`

Summary of findings:

- TF-IDF produced focused and interpretable recommendations.
- KNN produced relevant recommendations with slightly more variety.
- TF-IDF was faster during recommendation generation.
- TF-IDF and KNN had 60% overlap for the Python and Business/Finance examples, and 40% overlap for the Web Development example.
- The best text representation from the ablation study was `course_title + subject + level`.

## Graph Outputs

Presentation graph files are stored in:

```text
results/graphs/
```

The graph summary file explains each exported graph:

```text
results/graphs/graph_summary.md
```

## Synthetic Learner Personas

Synthetic learner profiles are stored in:

```text
data/synthetic_users.csv
```

These personas are for demonstration purposes only. They are not real users and are not used for collaborative filtering. They support interface personalization demos in the Streamlit app.

## Streamlit Dashboard

The dashboard is implemented in:

```text
app.py
```

It supports two modes:

1. Course-Based Recommendation
2. Learner Persona Recommendation

The learner persona mode uses synthetic profiles and simple matching-based recommendations for demonstration.

Run the dashboard with:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended Workflow

Run the notebooks in order:

```text
01_data_understanding.ipynb
02_preprocessing.ipynb
03_tfidf_recommender.ipynb
04_knn_recommender.ipynb
05_evaluation.ipynb
```

Then launch the dashboard:

```bash
streamlit run app.py
```

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit

## Project Status

The project includes:

- completed data understanding notebook
- completed preprocessing notebook
- completed TF-IDF recommender notebook
- completed KNN recommender notebook
- completed evaluation notebook
- evaluation result exports
- graph exports
- synthetic learner personas
- Streamlit dashboard prototype

## License

This project is licensed under the terms in the `LICENSE` file.
