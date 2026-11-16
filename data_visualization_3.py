import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load the Iris Dataset
iris = sns.load_dataset('iris')

# STEP 1a: List down features and their types
print("--- 1. Features and Their Types ---")
for col in iris.columns:
    feature_type = "Nominal (Categorical)" if iris[col].dtype == 'object' or str(iris[col].dtype) == 'category' else "Numeric (Continuous)"
    print(f"Feature: {col:15} | Data Type: {str(iris[col].dtype):10} | Statistical Type: {feature_type}")

# STEP 2: Create a Histogram for each numeric feature
iris.hist(figsize=(10, 8), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('2. Histograms for Each Feature in Iris Dataset', fontsize=14)
plt.tight_layout()
plt.show()

# STEP 3: Create a Boxplot for each numeric feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris.drop('species', axis=1), palette='Set3')
plt.title('3. Boxplots for Each Feature in Iris Dataset')
plt.ylabel('Measurement (in cm)')
plt.show()