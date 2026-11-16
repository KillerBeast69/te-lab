import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# STEP 1: Load Iris Dataset
df = sns.load_dataset('iris')
# Or: df = pd.read_csv('iris.csv')

print("--- Iris Dataset Preview ---")
print(df.head())

# STEP 2: Separate Features (X) and Target (y)
X = df.drop('species', axis=1)
y = df['species']

# STEP 3: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train Gaussian Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# STEP 5: Predict on Test Data
y_pred = nb_model.predict(X_test)

# STEP 6: Compute Confusion Matrix (3x3 Matrix because Iris has 3 classes)
cm = confusion_matrix(y_test, y_pred)
print("\n--- 3x3 Confusion Matrix ---")
print(cm)

# Calculate TP, FP, FN, TN for each class in the 3x3 matrix
classes = nb_model.classes_
print("\n--- Class-wise TP, FP, FN, TN ---")
for i, class_name in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    print(f"Species '{class_name}': TP={TP}, FP={FP}, FN={FN}, TN={TN}")

# Overall Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("\n--- Overall Evaluation Metrics ---")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")