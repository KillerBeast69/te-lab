import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load the inbuilt Titanic dataset
titanic = sns.load_dataset('titanic')
print("--- Titanic Dataset Preview ---")
print(titanic.head())

# Set visual theme
sns.set_theme(style="whitegrid")

# STEP 2: Find Patterns in the Data using Seaborn
plt.figure(figsize=(12, 5))

# Pattern 1: Survival Count based on Gender
plt.subplot(1, 2, 1)
sns.countplot(x='survived', hue='sex', data=titanic, palette='Set2')
plt.title('Pattern 1: Survival Count by Gender (0=Died, 1=Survived)')

# Pattern 2: Survival Count based on Passenger Class (1st, 2nd, 3rd Class)
plt.subplot(1, 2, 2)
sns.countplot(x='pclass', hue='survived', data=titanic, palette='viridis')
plt.title('Pattern 2: Survival Count by Passenger Class')

plt.tight_layout()
plt.show()

# STEP 3: Plot a Histogram to check how Ticket Price ('fare') is Distributed
plt.figure(figsize=(8, 5))
sns.histplot(titanic['fare'], bins=30, kde=True, color='teal')
plt.title('Distribution of Titanic Ticket Fare')
plt.xlabel('Fare (Ticket Price)')
plt.ylabel('Number of Passengers')
plt.show()