import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Step 1: Download/Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
if not os.path.exists("dataset.csv"):
    import urllib.request
    urllib.request.urlretrieve(url, "dataset.csv")

df = pd.read_csv("dataset.csv", sep=";")

print(f"Shape of dataset: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Create target variable: good (1) if quality >= 7, else not good (0)
df['target'] = (df['quality'] >= 7).astype(int)

# Class counts
print(f"Class counts:\n{df['target'].value_counts()}")

# Preprocess:
# Check missing values and duplicates
print(f"Missing values:\n{df.isnull().sum()}")
duplicates = df.duplicated().sum()
print(f"Duplicates: {duplicates}")

# Drop duplicates if any
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")

# Features and target
X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# Output min and max of each feature for Streamlit sliders
print("Feature ranges:")
for col in X.columns:
    print(f"{col}: min={X[col].min()}, max={X[col].max()}, mean={X[col].mean()}")

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Plot 1: Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Class Distribution (0 = Not Good, 1 = Good)')
plt.savefig('plots/class_distribution.png')
plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Models to evaluate
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]),
    'Decision Tree': Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=42))])
}

results = {}
best_model_name = None
best_score = 0
best_model = None

print("\nEvaluating models using 5-Fold Cross Validation:")
for name, model in models.items():
    # Evaluate with Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    mean_cv_score = cv_scores.mean()
    print(f"{name} - Mean CV F1 Score: {mean_cv_score:.4f}")
    
    # Train on full training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Selecting best model based on F1 score on cross-validation
    if mean_cv_score > best_score:
        best_score = mean_cv_score
        best_model_name = name
        best_model = model

print(f"\nBest Model: {best_model_name}")

# Save the best model
joblib.dump(best_model, "model.pkl")

# Evaluation of best model
bm_results = results[best_model_name]
print(f"Test Set Performance of {best_model_name}:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
    print(f"{metric}: {bm_results[metric]:.4f}")

# Plot 3: Confusion Matrix of Best Model
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, bm_results['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Good', 'Good'], yticklabels=['Not Good', 'Good'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Plot 4: ROC Curve of Best Model
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, bm_results['y_prob'])
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {bm_results['ROC AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('plots/roc_curve.png')
plt.close()

# Plot 5: Model Comparison Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
model_names = list(results.keys())
scores = {metric: [results[model][metric] for model in model_names] for metric in metrics}

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(model_names):
    ax.bar(x + i*width, [results[model][m] for m in metrics], width, label=model)

ax.set_ylabel('Scores')
ax.set_title('Model Comparison on Test Set')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
plt.ylim(0, 1.1)
plt.savefig('plots/model_comparison.png')
plt.close()

print("All tasks completed. Models evaluated and plots saved.")
