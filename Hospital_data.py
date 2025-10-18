import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

df = pd.read_csv("E:\\PGDM\\Data analytics\\Datasets\\Hospital Patient.csv")

print(df.head())
print(df.isnull().sum())

df.info()
print(df.describe())

### ------------------------- Exploratory Data Analysis (EDA) ------------------------- ###
# Unique values in each column

for col in df.columns:
    print(col, df[col].nunique())

# Total patients
total_patients = df.shape[0]

# Gender split
gender_split = (df['Gender'].value_counts(normalize=True) * 100).round(2)

# Average age
average_age = df['Age'].mean()

# Average treatment cost
avg_treatment_cost = df['Total_Cost'].mean()

# % of patients readmitted
df['Readmission_numeric'] = df['Readmission'].map({'Yes': 1, 'No': 0})
readmission_rate = (df['Readmission_numeric'].mean()) * 100

# Display summary
print("------ Business Summary ------")
print(f"Total Patients: {total_patients}")
print(f"Gender Split (%):\n{gender_split}")
print(f"Average Age: {average_age:.1f} years")
print(f"Average Treatment Cost: ₹{avg_treatment_cost:,.2f}")
print(f"Readmission Rate: {readmission_rate:.2f}%")

### Data Cleaning ###
df.drop(columns=['Readmission_numeric'], inplace=True)
df.dropna(inplace=True)
print(df.isnull().sum())
### Data Transformation ###
df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
df['Length_of_Stay'] = (df['Discharge_Date'] - df['Admission_Date']).dt.days
print(df[['Admission_Date', 'Discharge_Date', 'Length_of_Stay']].head())


### -------------------------------------- Data Visualization ----------------------------------- ###


# # Gender Distribution
# plt.figure(figsize=(6,4))
# sns.countplot(x='Gender', data=df)
# plt.title('Gender Distribution')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

# # Age Distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df['Age'], bins=20, kde=True)
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.show()

# # Treatment Cost Distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df['Total_Cost'], bins=30, kde=True)
# plt.title('Treatment Cost Distribution')
# plt.xlabel('Total Cost')
# plt.ylabel('Count')
# plt.show()

# # Readmission Rates by Condition
# # plt.figure(figsize=(10,6))
# # sns.barplot(x='Condition', y='Readmission_Rate', data=df)
# # plt.title('Readmission Rates by Condition')
# # plt.xlabel('Condition')
# # plt.ylabel('Readmission Rate (%)')
# # plt.xticks(rotation=45)
# # plt.show()

# # Length of Stay Distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df['Length_of_Stay'], bins=30, kde=True)
# plt.title('Length of Stay Distribution')
# plt.xlabel('Length of Stay (days)')
# plt.ylabel('Count')
# plt.show()

# # Average Treatment Cost by Condition
# plt.figure(figsize=(10,6))
# sns.barplot(x='Condition', y='Total_Cost', data=df)
# plt.title('Average Treatment Cost by Condition')
# plt.xlabel('Condition')
# plt.ylabel('Average Treatment Cost (₹)')
# plt.xticks(rotation=45)
# plt.show()


# # Set style for better-looking plots
# sns.set_style("whitegrid")
# plt.rcParams['figure.facecolor'] = 'white'


# 1. Gender Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Gender Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add count labels on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=10)
plt.tight_layout()
plt.show()


# 2. Average Treatment Cost by Condition
plt.figure(figsize=(12, 6))
cost_by_condition = df.groupby('Condition')['Total_Cost'].mean().sort_values(ascending=False)
ax = sns.barplot(x=cost_by_condition.index, y=cost_by_condition.values, palette='viridis')
plt.title('Average Treatment Cost by Condition', fontsize=14, fontweight='bold')
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Average Treatment Cost (₹)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for i, v in enumerate(cost_by_condition.values):
    ax.text(i, v + max(cost_by_condition.values)*0.01, f'₹{v:,.0f}', 
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# 3. Year of admission by total cost
df['Admission_Year'] = df['Admission_Date'].dt.year
plt.figure(figsize=(10, 6))
cost_by_year = df.groupby('Admission_Year')['Total_Cost'].mean()
ax = sns.lineplot(x=cost_by_year.index, y=cost_by_year.values, marker='o')
plt.title('Average Treatment Cost by Year of Admission', fontsize=14, fontweight='bold')
plt.xlabel('Year of Admission', fontsize=12)
plt.ylabel('Average Treatment Cost (₹)', fontsize=12)

# Add value labels on points
for i, v in enumerate(cost_by_year.values):
    ax.text(i, v + max(cost_by_year.values)*0.01, f'₹{v:,.0f}', 
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

### ----------------------------------------- Summary of visualizations ---------------------------------------- ###

print("\n📊 Visualization Summary:")
print(f"Total Records: {len(df):,}")
print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Average Cost: ₹{df['Total_Cost'].mean():,.2f}")
print(f"Average Stay: {df['Length_of_Stay'].mean():.1f} days")



