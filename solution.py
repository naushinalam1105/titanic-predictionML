import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# load dataset
df = pd.read_csv("titanic.csv")

# basic inspection
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# encoding categorical columns
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])

ohe = OneHotEncoder(sparse_output=False, drop='first')
embarked_encoded = ohe.fit_transform(df[['Embarked']])
embarked_cols = ohe.get_feature_names_out(['Embarked'])

embarked_df = pd.DataFrame(embarked_encoded, columns=embarked_cols, index=df.index)
df = pd.concat([df, embarked_df], axis=1)

# plot age distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# save cleaned output
df.to_csv("titanic_cleaned.csv", index=False)
print("Saved to titanic_cleaned.csv")