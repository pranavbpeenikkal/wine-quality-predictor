# Wine Quality Prediction Report

## 1. Problem Statement
The objective of this project is to develop a machine learning model to predict the quality of red wine based on its chemical properties. The problem is framed as a binary classification task, where wine with a quality score of 7 or higher is considered "good" (target = 1), and all other wines are considered "not good" (target = 0). This will help in automating the assessment of wine quality, ensuring consistency and saving time compared to human sensory evaluation. 

## 2. Dataset Description
The dataset used is the "Wine Quality (Red)" dataset from the UCI Machine Learning Repository.
* **Rows (Samples):** 1599 (before duplicate removal)
* **Features:** 11 chemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
* **Classes:** 2 (Target: 0 = Not Good, 1 = Good). Approximately 13.5% of the wines in the dataset have a quality of 7 or higher.

## 3. Preprocessing
The preprocessing pipeline involved the following steps:
1. **Handling Missing Values & Duplicates:** The dataset was checked for missing values and duplicates. Duplicates were identified and removed to prevent data leakage and bias in the training set.
2. **Text Columns:** There are no text columns in the Red Wine Quality dataset, so no encoding (like One-Hot Encoding) was necessary.
3. **Feature Scaling:** Since the features have different scales (e.g., density vs. total sulfur dioxide), we used `StandardScaler` to standardize the features (mean = 0, standard deviation = 1). This is crucial for models like SVM and Logistic Regression. To avoid data leakage, the scaler was applied within a `Pipeline`.
4. **Data Splitting:** The dataset was split into 80% training and 20% testing sets using `train_test_split`. We used `stratify=y` to ensure that both the training and testing sets have the same proportion of "good" and "not good" wines, which is important given the class imbalance.

## 4. Algorithm Selection
We selected three diverse machine learning algorithms for comparison:
1. **Logistic Regression:** Selected as a strong, interpretable linear baseline. It helps establish whether a simple linear combination of chemical properties can accurately predict wine quality.
2. **Support Vector Machine (SVM):** Selected for its ability to model complex, non-linear decision boundaries using the kernel trick, which may capture intricate relationships between chemical compounds and wine quality.
3. **Decision Tree:** Selected to test a non-linear, tree-based approach. It can handle non-linear feature interactions well and is highly interpretable.

## 5. Training and Testing
The models were incorporated into a `Pipeline` along with the `StandardScaler`. We evaluated the models using 5-Fold Cross-Validation on the training set to choose the best model. We focused on the F1-score during cross-validation due to the class imbalance (many more "not good" wines than "good" wines). 

The best performing model in cross-validation was chosen and trained on the full 80% training set (`.fit()`). It was then evaluated on the hold-out 20% test set using `.predict()` and `.predict_proba()`.

## 6. Metrics Table
*(Results from the test set evaluation)*

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8750 | 0.6000 | 0.2432 | 0.3462 | 0.8860 |
| SVM | 0.8787 | 0.6190 | 0.3514 | 0.4483 | 0.8845 |
| Decision Tree | 0.8235 | 0.3864 | 0.4595 | 0.4198 | 0.6698 |

*Logistic Regression emerged as the best model based on cross-validation F1-score.*

## 7. Graphs
*See the attached `plots/` folder for all generated graphs.*
1. **Class Distribution:** `plots/class_distribution.png` - Highlights the class imbalance (Target 0 vs Target 1).
2. **Correlation Heatmap:** `plots/correlation_heatmap.png` - Shows correlations between features (e.g., alcohol positively correlates with quality, while volatile acidity negatively correlates).
3. **Confusion Matrix:** `plots/confusion_matrix.png` - Details the True Positives, False Positives, True Negatives, and False Negatives of the best model (Logistic Regression).
4. **ROC Curve:** `plots/roc_curve.png` - Visualizes the trade-off between the True Positive Rate and False Positive Rate at different classification thresholds.
5. **Model Comparison:** `plots/model_comparison.png` - A bar chart comparing all models across the five evaluated metrics.

## 8. Analysis
* **Which model won:** Logistic Regression outperformed the other models (SVM, Decision Tree) on cross-validation F1-score, establishing itself as a strong linear baseline for this dataset. This suggests that the relationship between the chemical properties and wine quality can be reasonably approximated linearly.
* **What the errors mean:**
    * **False Positives (Type I Error):** Predicting a wine is "Good" when it is actually "Not Good". This is a false alarm.
    * **False Negatives (Type II Error):** Predicting a wine is "Not Good" when it is actually "Good". This is a missed opportunity.
* **Which metric matters most:** For a winery, a False Positive (selling a bad wine labeled as premium/good) hurts the brand's reputation significantly. A False Negative (missing out on labeling a good wine as premium) is a lost opportunity but less damaging to reputation. Therefore, **Precision** (minimizing False Positives) is arguably the most critical metric. However, since we still want to capture a decent amount of good wine, the **F1-score** (harmonic mean of Precision and Recall) provides a good balance, making it a reliable metric for model selection in this imbalanced scenario.
* **Limitations:** The dataset is relatively small (~1359 rows after dropping duplicates) and imbalanced. It also only covers a specific variant (Portuguese "Vinho Verde" red wine), so the model might not generalize well to white wines or wines from other regions.

## 9. Mapping to Course Outcome
**"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures"**
This project successfully maps to CO5. We identified a real-life problem (wine quality prediction), selected appropriate models (Logistic Regression, SVM, Decision Tree), applied preprocessing techniques (scaling, handling missing values/duplicates, stratification), and rigorously evaluated the models using multiple performance measures (Accuracy, Precision, Recall, F1, ROC-AUC) and visual evaluations (Confusion Matrices, ROC Curves, Heatmaps).
