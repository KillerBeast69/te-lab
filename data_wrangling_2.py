import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a realistic Academic Performance dataset with missing values and outliers
# (If you have a CSV file in the lab, you can also use: df = pd.read_csv('Academic_Performance.csv'))
data = {
    'Student_ID': range(1, 16),
    'Math_Score': [65, 70, np.nan, 75, 80, 85, 60, 250, 72, 68, np.nan, 77, 82, 5, 74], # 250 and 5 are outliers!
    'Reading_Score': [60, 65, 70, 72, 68, 75, np.nan, 80, 78, 82, 85, 88, 90, 62, 71],
    'Attendance_Hours': [10, 12, 15, 11, 14, 13, 18, 20, 45, 90, 120, 200, 15, 16, 14] # Skewed data
}
df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# STEP 1: Scan and Deal with Missing Values
print("\n--- Missing Values Before Imputation ---")
print(df.isnull().sum())

# Technique: Fill missing numerical values with the Median (robust against outliers) or Mean
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].median())
df['Reading_Score'] = df['Reading_Score'].fillna(df['Reading_Score'].mean())

print("\n--- Missing Values After Imputation ---")
print(df.isnull().sum())

# STEP 2: Scan Numeric Variables for Outliers and Deal with Them
# Using the Interquartile Range (IQR) Method on 'Math_Score'
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR = {IQR}, Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

# Identify outliers
outliers = df[(df['Math_Score'] < lower_bound) | (df['Math_Score'] > upper_bound)]
print("\n--- Detected Outliers in Math_Score ---")
print(outliers[['Student_ID', 'Math_Score']])

# Deal with outliers by Capping/Trimming them to the upper and lower bounds
df['Math_Score'] = np.where(df['Math_Score'] > upper_bound, upper_bound,
                   np.where(df['Math_Score'] < lower_bound, lower_bound, df['Math_Score']))

print("\n--- Math_Score After Treating Outliers ---")
print(df['Math_Score'].values)

# STEP 3: Apply Data Transformation
# Purpose: Decrease skewness in 'Attendance_Hours' using Log Transformation
print(f"\nSkewness BEFORE Log Transformation: {df['Attendance_Hours'].skew():.2f}")

df['Log_Attendance'] = np.log10(df['Attendance_Hours'])

print(f"Skewness AFTER Log Transformation: {df['Log_Attendance'].skew():.2f}")
print("\n--- Final Transformed DataFrame ---")
print(df.head())