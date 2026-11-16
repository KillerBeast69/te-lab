import pandas as pd
import numpy as np
import seaborn as sns

# ==========================================
# PART 1: Summary Statistics Grouped by a Categorical Variable
# ==========================================
data = {
    'Age_Group': ['Young', 'Young', 'Middle', 'Middle', 'Senior', 'Senior', 'Young', 'Middle', 'Senior'],
    'Age': [22, 25, 38, 42, 61, 65, 28, 45, 70],
    'Income': [30000, 35000, 65000, 72000, 50000, 48000, 40000, 80000, 52000]
}
df_income = pd.DataFrame(data)

# Group by categorical variable 'Age_Group' and calculate summary statistics for 'Income'
summary_stats = df_income.groupby('Age_Group')['Income'].agg(['mean', 'median', 'min', 'max', 'std'])
print("--- PART 1: Summary Statistics of Income Grouped by Age_Group ---")
print(summary_stats)

# Create a list containing a numeric value for each response to the categorical variable
mapping = {'Young': 1, 'Middle': 2, 'Senior': 3}
numeric_category_list = df_income['Age_Group'].map(mapping).tolist()
print("\nNumeric List for Categorical Responses:", numeric_category_list)


# ==========================================
# PART 2: Statistical Details of Iris Dataset Species
# ==========================================
# Load Iris dataset (if you have iris.csv, use pd.read_csv('iris.csv'))
iris = sns.load_dataset('iris')

print("\n--- PART 2: Basic Statistical Details for Each Iris Species ---")

# Filter for each species and display .describe() (which shows mean, std, min, max, and 25%, 50%, 75% percentiles)
for species_name in ['setosa', 'versicolor', 'virginica']:
    print(f"\n********** Statistics for Iris-{species_name} **********")
    species_df = iris[iris['species'] == species_name]
    print(species_df.describe())