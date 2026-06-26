# Manish Pawar - Project 2 Task 2
# Random Forest + Evaluation on Iris Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = target_names[y]
print('Dataset shape:', df.shape)
print('Species:', df['species'].unique())
print('\nFirst 5 rows:')
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print('\nTraining samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('\nRandom Forest Accuracy:', round(accuracy_score(y_test, y_pred_rf)*100, 2), '%')

dt_cv = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X, y, cv=5, scoring='accuracy')
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1), X, y, cv=5, scoring='accuracy')
print('\nDecision Tree CV Accuracy:')
print('Each fold:', np.round(dt_cv, 3))
print(f'Mean: {dt_cv.mean():.3f} | Std: {dt_cv.std():.3f}')
print('\nRandom Forest CV Accuracy:')
print('Each fold:', np.round(rf_cv, 3))
print(f'Mean: {rf_cv.mean():.3f} | Std: {rf_cv.std():.3f}')

# Feature importance
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(4), importances[sorted_idx], edgecolor='black')
ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=15)
ax.set_title('Feature Importance - Random Forest (Iris)', fontweight='bold')
ax.set_ylabel('Importance Score')
for bar, imp in zip(bars, importances[sorted_idx]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout(); plt.savefig('images/feature_importance.png', dpi=150); plt.close()
print('\nSaved: images/feature_importance.png')

models = {
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=300, random_state=42),
}
results = {}
print('\nModel Comparison using 5-Fold Cross-Validation:')
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f'{name}: {scores.mean():.3f} +/- {scores.std():.3f}')
fig, ax = plt.subplots(figsize=(11, 6))
names = list(results.keys())
means = [results[n].mean() for n in names]
stds = [results[n].std() for n in names]
bars = ax.barh(names, means, xerr=stds, edgecolor='black', height=0.6, error_kw={'capsize':5})
for bar, mean in zip(bars, means):
    ax.text(mean+0.002, bar.get_y()+bar.get_height()/2, f'{mean:.3f}', va='center', fontweight='bold')
ax.set_xlabel('Cross-Validation Accuracy (mean +/- std)')
ax.set_title('Model Comparison - Iris Dataset\n(Soft Nexis Technology Internship)', fontweight='bold')
ax.set_xlim(0.85, 1.05)
plt.tight_layout(); plt.savefig('images/model_comparison.png', dpi=150); plt.close()
print('Saved: images/model_comparison.png')

# Hyperparameter tuning. Small grid used so it runs quickly in internship submission.
param_grid = {'n_estimators':[50,100], 'max_depth':[3,5,None], 'min_samples_split':[2,5]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)
print('\nBest Parameters:', grid_search.best_params_)
print('Best CV Accuracy:', round(grid_search.best_score_, 4))
best_model = grid_search.best_estimator_
y_final_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_final_pred)
print(f'Final Test Accuracy with best model: {final_accuracy*100:.2f}%')

print('\n' + '='*60)
print('FINAL MODEL EVALUATION REPORT')
print('Soft Nexis Technology - ML Internship')
print('='*60)
print('Student Name : Manish Pawar')
print('Algorithm    : Random Forest (Tuned)')
print(f'Test Accuracy: {final_accuracy*100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_final_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_final_pred)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_title('Final Random Forest - Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('images/final_confusion_matrix.png', dpi=150); plt.close()
print('Saved: images/final_confusion_matrix.png')
