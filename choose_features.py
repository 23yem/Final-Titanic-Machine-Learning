import pandas as pd
from scipy.stats import chi2_contingency

# Load the data
df = pd.read_csv('train.csv')

# Create a cross-tabulation of 'Embarked' and 'Survived'
contingency_table = pd.crosstab(df['Embarked'], df['Survived'])

# Perform the chi-square test for independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the p-value
print(p) # 1.769922284120912e-06, so there might be some correleation


# Get unique values
unique_values = df['Embarked'].unique()
print("Unique values in 'Embarked':", unique_values)

# Get number of unique values
num_unique_values = df['Embarked'].nunique()
print("Number of unique values in 'Embarked':", num_unique_values)

#print(df["Embarked"].isnull().sum())

# Convert 'Embarked' to a categorical type
df['Embarked'] = df['Embarked'].astype('category')


print(df["Embarked"])

# # Create a dictionary that maps category codes to categories
# code_to_category_dict = dict(enumerate(df['Embarked'].cat.categories))

# print(code_to_category_dict)

# # Convert categories to codes
# df['Embarked'] = df['Embarked'].cat.codes