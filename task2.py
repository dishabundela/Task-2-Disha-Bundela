import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("Original Dataset Shape:")
print(df.shape)

# Missing Values Check
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Age with Mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill Missing Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin Column
df = df.drop("Cabin", axis=1)

# Remove Duplicates
df = df.drop_duplicates()

# Label Encoding
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Standardization
scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# Save Cleaned Dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nCleaning Completed Successfully!")
print("\nNew Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())