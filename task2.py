#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importDS
import kagglehub
import os
path = kagglehub.dataset_download("brendan45774/test-file")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
print(df.head())

#basicInfo
print(df.info())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)

#summeryStatistics
print(df.describe())

#checkForMissingValue
print(df.isnull().sum())

#histograms
df.hist(figsize=(12,10))
plt.tight_layout()
plt.savefig("histograms.png", dpi=300)
plt.show()

#ageDistribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png", dpi=300)
plt.show()

#boxplotForOutliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.savefig("boxplot_fare.png", dpi=300)
plt.show()

#survivalCount
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png", dpi=300)
plt.show()

#survivalByGender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_gender.png", dpi=300)
plt.show()

#corrMatrix
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].fillna('Unknown'))

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

#pairPlot
sns.pairplot(df[['Age','Fare','Pclass','Survived']])
plt.savefig("pairplot.png", dpi=300)
plt.show()