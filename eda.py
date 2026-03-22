# ================================
# TITANIC SURVIVAL PREDICTION - EDA & MODEL
# ================================

# -------- IMPORT LIBRARIES --------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# -------- LOAD DATA --------
df = pd.read_csv("titanic.csv")


# ================================
# 🔍 EXPLORATORY DATA ANALYSIS (EDA)
# ================================

print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- FIRST 10 ROWS ---")
print(df.head(10))

print("\n--- LAST 10 ROWS ---")
print(df.tail(10))

print("\n--- COLUMN NAMES ---")
print(df.columns)

print("\n--- DATA SHAPE ---")
print(df.shape)

print("\n--- DUPLICATES ---")
print("Duplicate rows:", df.duplicated().sum())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())


# -------- VISUALIZE MISSING DATA --------
ms.bar(df, figsize=(10, 5), color="tomato")
plt.title("Missing Data Visualization", size=15, color="red")
plt.show()


# ================================
# 🧹 DATA CLEANING
# ================================

# Drop 'Cabin' column (too many missing values)
df.drop(['Cabin'], axis=1, inplace=True)

print("\nAfter dropping Cabin column:", df.shape)


# -------- HANDLE 'Embarked' COLUMN --------
print("\nEmbarked Unique Values:", df["Embarked"].unique())
print(df["Embarked"].value_counts())

# Fill missing values with most frequent value ('S')
df["Embarked"] = df["Embarked"].fillna("S")

print("\nAfter filling Embarked:")
print(df["Embarked"].value_counts())


# -------- HANDLE 'Age' COLUMN --------
print("\nAge Statistics:")
print("Mean:", df["Age"].mean())
print("Median:", df["Age"].median())
print("Mode:", df["Age"].mode())

# Fill missing values with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

print("Missing Age values:", df["Age"].isnull().sum())


# -------- CHECK AGAIN AFTER CLEANING --------
ms.bar(df, figsize=(10, 5), color="tomato")
plt.title("Missing Data After Cleaning", size=15, color="red")
plt.show()

print("\nFinal Missing Values:")
print(df.isnull().sum())


# ================================
# 📊 DATA VISUALIZATION
# ================================

# Survival count
print("\nSurvival Count:")
print(df["Survived"].value_counts())

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()


# Gender distribution
print("\nGender Count:")
print(df["Sex"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(6, 4))

# Pie chart
df["Sex"].value_counts().plot(
    kind="pie",
    ax=axes[0],
    autopct='%0.1f%%',
    colormap="Reds"
)

# Bar chart
df["Sex"].value_counts().plot(
    kind="bar",
    ax=axes[1],
    color=['darkred', 'indianred']
)

plt.show()


# Gender vs Survival
sns.catplot(x="Sex", hue="Survived", kind="count", data=df, height=4)
plt.title("Gender vs Survival")
plt.show()


# Passenger Class vs Survival
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Reds")
plt.title("Class vs Survival")
plt.show()


# ================================
# 🔄 FEATURE ENGINEERING
# ================================

# Drop unnecessary columns
df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# Convert categorical data into numerical
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())


# Replace any remaining NaN values (safety step)
df['Age'] = df['Age'].replace(np.nan, 0)
df['Embarked'] = df['Embarked'].replace(np.nan, 0)


# ================================
# 🤖 MODEL BUILDING
# ================================

# Split features and target
X = df.drop(["Survived"], axis=1)
y = df["Survived"]

print("\nTarget Variable (y):")
print(y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

print("\n--- DATA SPLIT ---")
print("Total Data:", df.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ================================
# 🚀 MODEL TRAINING (Naive Bayes)
# ================================

model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nPredictions:", y_pred)
print("Actual:", y_test.values)


# ================================
# 📈 MODEL EVALUATION
# ================================

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ MODEL ACCURACY:", accuracy)