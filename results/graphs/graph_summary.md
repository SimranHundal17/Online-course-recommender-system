# Graph Summary

These graphs support the final presentation using outputs from the project notebooks. The data-understanding graphs were exported from existing notebook chart outputs, and the evaluation graphs visualize results already present in `notebooks/05_evaluation.ipynb`.

## Exported Graphs

### `number_of_courses_per_subject_category.png`

- Source notebook: `notebooks/01_data_understanding.ipynb`
- Source section: `Number of Courses per Subject/Category`
- Purpose: Shows how courses are distributed across Udemy subject categories. This is useful in the final presentation for explaining dataset composition and subject coverage.

### `distribution_of_subscribers.png`

- Source notebook: `notebooks/01_data_understanding.ipynb`
- Source section: `Distribution of Subscribers`
- Purpose: Shows the distribution of subscriber counts across courses. This supports the data understanding discussion by highlighting course popularity patterns and subscriber skew.

### `distribution_of_course_levels.png`

- Source notebook: `notebooks/01_data_understanding.ipynb`
- Source section: `Distribution of Course Levels`
- Purpose: Shows the number of courses available at each difficulty level. This is useful for explaining learner-level coverage in the dataset.

## Evaluation Graphs

### `recommendation_overlap_comparison.png`

This graph shows the percentage of shared Top-5 recommendations between TF-IDF and KNN for the Python-related, Business/Finance, and Web Development examples. It matters because overlap indicates how much the two recommenders agree on relevant courses, while lower overlap can suggest that one method is introducing more varied alternatives.

### `runtime_comparison_evaluation.png`

This graph compares average recommendation generation time for TF-IDF and KNN using the runtime values reported in the evaluation notebook. It matters because runtime helps compare practical usability: TF-IDF generated recommendations faster, while KNN remained acceptable for the project dataset size.

### `ablation_study_comparison.png`

This graph compares Top-5 similarity scores for the ablation experiments: title only, title + subject, and title + subject + level. It matters because it shows how adding subject and level information changes recommendation scoring and supports the final decision to use the `title + subject + level` representation.

## Notebook Export Notes

- `notebooks/01_data_understanding.ipynb` contained 3 embedded PNG chart outputs, and all 3 were exported.
- `notebooks/05_evaluation.ipynb` did not contain embedded chart image outputs. Its evaluation results are represented as markdown tables, so the evaluation graphs in this folder directly visualize those existing reported results without adding new metrics.
