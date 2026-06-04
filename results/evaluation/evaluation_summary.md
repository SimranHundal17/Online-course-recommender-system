# Evaluation Summary

## TF-IDF Findings

The TF-IDF recommender produced highly focused and interpretable recommendations. It performed especially well when the input course contained distinctive terms such as `python`, `investment banking`, and `web development`. It was also the faster model during recommendation generation, averaging `0.001245` seconds over 100 runs for `web programming with python`.

## KNN Findings

The KNN recommender also produced relevant recommendations across the Python, Business/Finance, and Web Development examples. Its results were sometimes more varied because it combined text features with numeric course metadata. KNN averaged `0.028309` seconds over 100 recommendation runs for the same test course.

## Overlap Analysis

TF-IDF and KNN shared 3 of 5 recommendations for the Python-related example, giving 60% overlap. They also shared 3 of 5 recommendations for the Business/Finance example, giving 60% overlap. For the Web Development example, they shared 2 of 5 recommendations, giving 40% overlap. This suggests both models capture core topic similarity, while KNN can introduce more varied alternatives.

## Runtime Comparison

TF-IDF was faster during recommendation generation. KNN had acceptable performance for the dataset size of 3,672 courses, but the measured recommendation generation time was higher than TF-IDF. For this project dataset, both runtimes are practical.

## Ablation Study Conclusions

The ablation study used `web programming with python`. The `course_title only` experiment already performed well because the title contained strong keywords. Adding `subject` reinforced category relevance. Adding `level` provided useful difficulty context. The final recommended text representation is `course_title + subject + level`, matching the project's `combined_features` column.
