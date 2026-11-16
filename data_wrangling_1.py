# STEP 1: Import all required Python libraries
import pandas as pd
import numpy as np
import seaborn as sns

# STEP 2 & 3: Locate and Load the Dataset into a Pandas DataFrame
# Source URL: https://www.kaggle.com/c/titanic/data (or inbuilt via seaborn)
# Description: Contains passenger details (age, sex, fare, class, survival) of the Titanic ship.
df = sns.load_dataset('titanic')
# Note: If your examiner gives you a local file named 'data.csv', use:
# df = pd.read_csv('data.csv')

print("--- First 5 Rows of Dataset ---")
print(df.head())

# STEP 4: Data Preprocessing
# 4a. Check for missing values in each column
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 4b. Get initial statistical details (count, mean, std, min, percentiles, max)
print("\n--- Initial Statistics ---")
print(df.describe())

# 4c. Check the dimensions (rows, columns) of the DataFrame
print("\n--- Dimensions of DataFrame (Rows, Columns) ---")
print(df.shape)

# 4d. Check variable descriptions and data types
print("\n--- Data Types and Variable Info ---")
print(df.info())

# STEP 5: Data Formatting and Normalization (Type Conversion)
# Let's check current data types
print("\n--- Data Types Before Conversion ---")
print(df.dtypes)

# Suppose 'fare' is float64, or 'survived' is int64 and we want to convert 'pclass' to Category/String
# Let's fill missing values in 'age' first so we can convert 'age' from float to integer
df['age'] = df['age'].fillna(df['age'].mean())
df['age'] = df['age'].astype('int64')

print("\n--- Data Type of 'age' After Conversion to Integer ---")
print(df['age'].dtype)

# STEP 6: Turn Categorical Variables into Quantitative (Numeric) Variables
# Method 1: Replacing categories directly (Label Encoding)
df['sex_numeric'] = df['sex'].map({'male': 0, 'female': 1})

# Method 2: One-Hot Encoding using pd.get_dummies()
embarked_dummies = pd.get_dummies(df['embarked'], prefix='embarked', dtype=int)
df = pd.concat([df, embarked_dummies], axis=1)

print("\n--- Categorical to Quantitative Conversion Result ---")
print(df[['sex', 'sex_numeric', 'embarked', 'embarked_C', 'embarked_Q', 'embarked_S']].head())