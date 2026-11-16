import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# STEP 1: Load Dataset (Loads CSV if present, otherwise builds sample Social_Network_Ads data)
try:
    df = pd.read_csv('Social_Network_Ads.csv')
except FileNotFoundError:
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, 200),
        'EstimatedSalary': np.random.randint(15000, 150000, 200),
        'Purchased': np.random.choice([0, 1], 200)
    })

print("--- First 5 Rows of Social_Network_Ads ---")
print(df.head())

# STEP 2: Select Features (Age, EstimatedSalary) and Target (Purchased)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# STEP 3: Train-Test Split (75% Train, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# STEP 4: Feature Scaling (Crucial for Logistic Regression because Salary is in thousands and Age is in tens)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 5: Train Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# STEP 6: Make Predictions
y_pred = classifier.predict(X_test)

# STEP 7: Compute Confusion Matrix and Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\n--- Confusion Matrix ---")
print(cm)
print(f"\nTrue Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Negatives (FN): {FN}")

print("\n--- Performance Metrics ---")
print(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")