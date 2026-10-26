
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from docx import Document
from docx.shared import Inches

# ---------------- 1. SETUP ----------------

sns.set_theme(style="whitegrid")
os.makedirs("output", exist_ok=True)

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("Loading Titanic dataset...")
df = pd.read_csv(URL)
original = df.copy()

# ---------------- 2. DATA ACQUISITION ----------------

print("\nDataset shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nColumn information:")
df.info()

print("\nSummary statistics:\n", df.describe(include="all").T)
print("\nMissing values:\n", df.isnull().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

# ---------------- 3. MISSING VALUES VISUALIZATION ----------------

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(9, 5))
if len(missing):
    sns.barplot(x=missing.values, y=missing.index)
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
else:
    plt.text(0.5, 0.5, "No missing values", ha="center")
plt.title("Missing Values Before Cleaning")
plt.tight_layout()
plt.savefig("output/01_missing_values.png", dpi=150)
plt.close()

# ---------------- 4. DATA CLEANING ----------------

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Remove duplicate passenger IDs, if any
df = df.drop_duplicates(subset="PassengerId", keep="first").copy()

# Preserve useful information from Cabin before dropping it
df["HasCabin"] = df["Cabin"].notna().astype(int)

# Cabin has too many missing values to impute reliably
df = df.drop(columns=["Cabin"])

# Convert numeric columns safely
for col in ["Age", "Fare", "SibSp", "Parch", "Pclass", "Survived"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Age: median by passenger class and sex
age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(age_medians)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked: fill missing values with most frequent category
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Fare: median is robust to extreme values
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Remove rows missing essential outcome or class information
df = df.dropna(subset=["Survived", "Pclass"])

# Create useful features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print("\nRows after cleaning:", len(df))
print("\nMissing values after cleaning:\n", df.isnull().sum())

df.to_csv("output/titanic_cleaned.csv", index=False)

# ---------------- 5. EDA VISUALIZATIONS ----------------

# Chart 2: Survival distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("output/02_survival_distribution.png", dpi=150)
plt.close()

# Chart 3: Survival rate by gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Sex", y="Survived", errorbar=None)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/03_survival_by_gender.png", dpi=150)
plt.close()

# Chart 4: Survival rate by passenger class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class (1 = First, 3 = Third)")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/04_survival_by_class.png", dpi=150)
plt.close()

# Chart 5: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=25, kde=True)
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig("output/05_age_distribution.png", dpi=150)
plt.close()

# Chart 6: Correlation heatmap
numeric_cols = [
    "Survived", "Pclass", "Age", "SibSp",
    "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"
]
plt.figure(figsize=(10, 7))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/06_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------- 6. INSIGHTS ----------------

overall_survival = df["Survived"].mean()
gender_rates = df.groupby("Sex")["Survived"].mean()
class_rates = df.groupby("Pclass")["Survived"].mean()
median_age = df["Age"].median()
median_fare = df["Fare"].median()

insights = [
    f"Overall, {overall_survival:.1%} of passengers in this dataset survived.",
    f"Median passenger age is {median_age:.1f} years.",
    f"Median ticket fare is {median_fare:.2f}.",
    "Survival rate by gender:\n" +
    gender_rates.map(lambda x: f"{x:.1%}").to_string(),
    "Survival rate by passenger class:\n" +
    class_rates.map(lambda x: f"{x:.1%}").to_string(),
    f"{df['IsAlone'].mean():.1%} of passengers were travelling alone.",
    "These are observed associations, not proof that any single factor caused survival."
]

print("\nKEY INSIGHTS")
for insight in insights:
    print("-", insight)

# ---------------- 7. AUTOMATIC WORD REPORT ----------------

doc = Document()
doc.add_heading("Titanic Dataset: Data Cleaning and EDA", 0)
doc.add_paragraph("Week 1 Internship Task | Data Acquisition, Cleaning and Exploratory Analysis")

doc.add_heading("1. Objective", level=1)
doc.add_paragraph(
    "The objective is to acquire a public dataset, assess data quality, "
    "clean missing and duplicate records, explore distributions and relationships, "
    "and communicate findings using visualizations."
)

doc.add_heading("2. Dataset and Source", level=1)
doc.add_paragraph(
    "Dataset: Titanic passenger dataset\n"
    "Source: DataScienceDojo public GitHub repository\n"
    f"URL: {URL}\n"
    f"Original rows: {len(original)}\n"
    f"Original columns: {len(original.columns)}"
)
doc.add_paragraph(
    "The dataset contains passenger details such as age, sex, ticket class, "
    "fare, family members aboard, embarkation port and survival outcome."
)

doc.add_heading("3. Tools Used", level=1)
doc.add_paragraph(
    "Python, Pandas, NumPy, Matplotlib, Seaborn and python-docx."
)

doc.add_heading("4. Initial Data Assessment", level=1)
doc.add_paragraph(
    f"Original dataset shape: {original.shape}\n"
    f"Exact duplicate rows: {original.duplicated().sum()}\n"
    f"Missing values by column:\n{original.isnull().sum().to_string()}"
)

doc.add_heading("5. Data Cleaning and Rationale", level=1)
cleaning_steps = [
    "Removed exact duplicate rows to avoid counting identical records more than once.",
    "Checked PassengerId for duplicate identifiers and retained the first occurrence.",
    "Created HasCabin before dropping Cabin because Cabin has substantial missingness.",
    "Converted relevant numeric columns with errors coerced to missing values.",
    "Filled missing Age values using median age for the same passenger class and sex, then overall median as fallback.",
    "Filled missing Embarked values with the most frequent category.",
    "Filled missing Fare values with the median.",
    "Removed records missing essential Survived or Pclass values.",
    "Created FamilySize and IsAlone features to support further analysis."
]
for step in cleaning_steps:
    doc.add_paragraph(step, style="List Bullet")

doc.add_paragraph(
    f"Final cleaned shape: {df.shape}\n"
    f"Remaining missing values: {int(df.isnull().sum().sum())}"
)

doc.add_heading("6. Visual Exploratory Analysis", level=1)
charts = [
    ("output/01_missing_values.png", "Figure 1. Missing values before cleaning."),
    ("output/02_survival_distribution.png", "Figure 2. Distribution of survival outcomes."),
    ("output/03_survival_by_gender.png", "Figure 3. Survival rate by gender."),
    ("output/04_survival_by_class.png", "Figure 4. Survival rate by passenger class."),
    ("output/05_age_distribution.png", "Figure 5. Passenger age distribution."),
    ("output/06_correlation_heatmap.png", "Figure 6. Correlations among numerical variables.")
]

for path, caption in charts:
    doc.add_paragraph(caption)
    doc.add_picture(path, width=Inches(5.8))

doc.add_heading("7. Key Findings", level=1)
for insight in insights:
    doc.add_paragraph(insight, style="List Bullet")

doc.add_heading("8. Limitations and Further Analysis", level=1)
doc.add_paragraph(
    "Median and mode imputation introduce assumptions into the data. "
    "Dropping Cabin loses individual cabin details, although HasCabin retains "
    "whether a cabin value was recorded. Correlation does not establish causation. "
    "Future work could investigate age groups, family size, ticket class and "
    "gender together, and test whether observed differences are statistically significant."
)

doc.add_heading("9. Conclusion", level=1)
doc.add_paragraph(
    "The dataset was inspected, cleaned and analyzed using descriptive statistics "
    "and visualizations. The resulting charts and insights provide an initial "
    "understanding of survival patterns and prepare the data for further analysis."
)

doc.save("output/Titanic_EDA_Report.docx")

print("\nSUCCESS! Check the output folder for:")
print("- Titanic_cleaned.csv")
print("- Titanic_EDA_Report.docx")
print("- Six PNG visualizations")
