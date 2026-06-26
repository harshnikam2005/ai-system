
# Task 1: Decision Tree Classifier on Iris Dataset
# Name: Manish Pawar
# Internship: Soft Nexis Technology - Data Science & ML in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]
print('Dataset shape:', df.shape)
print('Species:', df['species'].unique())
print('\nFirst five rows:')
print(df.head())

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print('\nTraining samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])
print('Training class distribution:', dict(zip(*np.unique(y_train, return_counts=True))))

# 3. Train Decision Tree
# max_depth=3 helps control model complexity and reduces overfitting.
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
print('\nModel training complete!')
print('Number of leaves in tree:', dt_model.get_n_leaves())
print('Tree depth:', dt_model.get_depth())

# 4. Make predictions
y_pred = dt_model.predict(X_test)
print('\nActual labels:   ', y_test[:10])
print('Predicted labels:', y_pred[:10])

new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = dt_model.predict(new_flower)
prediction_proba = dt_model.predict_proba(new_flower)
print('\nNew flower measurements:', new_flower[0])
print('Predicted species:', iris.target_names[prediction[0]])
print('Confidence:', prediction_proba)

# 5. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy*100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Save confusion matrix image
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax, cmap='Purples', values_format='d', colorbar=False)
ax.set_title('Decision Tree Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_dt.png', dpi=150)
plt.close()

# 6. Save decision tree visualization
fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title('Decision Tree - Iris Classifier\n(Soft Nexis Technology Internship)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_visual.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nDecision tree visualization saved as decision_tree_visual.png')

# 7. Experiment with tree depth to study overfitting
depths = [1, 2, 3, 4, 5, 10, None]
train_acc = []
test_acc = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))

results = pd.DataFrame({
    'max_depth': [str(d) for d in depths],
    'training_accuracy': train_acc,
    'test_accuracy': test_acc
})
print('\nDepth vs Accuracy Results:')
print(results)

fig, ax = plt.subplots(figsize=(9, 5))
depth_labels = [str(d) for d in depths]
ax.plot(depth_labels, train_acc, 'o-', label='Training Accuracy', linewidth=2)
ax.plot(depth_labels, test_acc, 's--', label='Test Accuracy', linewidth=2)
ax.set_title('Training vs Test Accuracy by Tree Depth', fontsize=13, fontweight='bold')
ax.set_xlabel('max_depth (None = unlimited)', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0.8, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_vs_accuracy.png', dpi=150)
plt.close()
print('Depth vs accuracy graph saved as depth_vs_accuracy.png')
