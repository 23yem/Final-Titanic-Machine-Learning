import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# Assuming 'df' is your DataFrame and it has columns 'Age' and 'Survived'
# Create age ranges

bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']
df['AgeRange'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calculate mean survival rate for each age range
age_survival = df.groupby('AgeRange')['Survived'].mean()

# Plot
age_survival.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Age Range')
plt.show()