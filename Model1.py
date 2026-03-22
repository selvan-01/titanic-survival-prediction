# ============================================
# TITANIC SURVIVAL PREDICTION - MODEL TRAINING
# ============================================

# -------- IMPORT LIBRARIES --------
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import pickle


# -------- LOAD DATA --------
df = pd.read_csv('titanic.csv')

print("\n--- DATA LOADED ---")
print(df.head())


# ============================================
# 🧹 DATA PREPROCESSING
# ============================================

# Encode categorical features (Sex, Embarked)
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])


# -------- HANDLE MISSING VALUES --------
# Replace null values with 0 (simple approach)
df['Age'] = df['Age'].replace(np.nan, 0)
df['Embarked'] = df['Embarked'].replace(np.nan, 0)

print("\n--- DATA AFTER PREPROCESSING ---")
print(df.head())


# ============================================
# 🎯 FEATURE SELECTION
# ============================================

# Drop unnecessary columns
X = df.drop(columns=['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'])

# Target variable
y = df['Survived']

print("\n--- FEATURES (X) ---")
print(X.head())

print("\n--- TARGET (y) ---")
print(y.head())


# ============================================
# 🔀 TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

print("\n--- DATA SPLIT ---")
print("Total Data:", df.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================
# 🤖 MODEL TRAINING (Naive Bayes)
# ============================================

model = GaussianNB()

# Train model
model.fit(X_train, y_train)

print("\n✅ Model Training Completed")


# ============================================
# 🔮 PREDICTION
# ============================================

y_pred = model.predict(X_test)

print("\n--- PREDICTIONS ---")
print("Predicted:", y_pred)
print("Actual:", y_test.values)


# ============================================
# 📊 MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 MODEL ACCURACY:", accuracy)


# ============================================
# 💾 SAVE MODEL
# ============================================

# Save trained model as .pkl file
pickle.dump(model, open('model.pkl', 'wb'))

print("\n💾 Model saved as 'model.pkl'")


# ============================================
# 🔁 SAMPLE PREDICTION (OPTIONAL)
# ============================================
"""
Example input format:
[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]

NOTE: Values must match preprocessing format
"""

# Example:
# sample = [[3, 1, 22, 1, 0, 7.25, 2]]
# prediction = model.predict(sample)

# if prediction == 1:
#     print("Passenger Survived")
# else:
#     print("Passenger Did Not Survive")