import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load the inbuilt Titanic dataset
titanic = sns.load_dataset('titanic')

# STEP 2: Plot Boxplot of Age with respect to Gender and Survival Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Pastel1')

plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender (Sex)')
plt.ylabel('Age of Passengers')
plt.legend(title='Survived (0=No, 1=Yes)')
plt.show()