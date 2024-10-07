import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.inspection import permutation_importance

# Load the dataset
file_path = '/Users/jaheergoulam/Desktop/Implementation Pyton/dataset/main_file_DR_selected_features_1-2.csv'
df = pd.read_csv(file_path)

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics of the dataset:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Distribution of the target variable
print("\nDistribution of the target variable 'NEW_CLASS':")
print(df['CLASE'].value_counts())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='CLASE', data=df)
plt.title('Distribution of the target variable (NEW_CLASS)')
plt.show()

# Visualize correlations
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation matrix of features')
plt.show()

# Preprocess the data
X = df.drop(columns=['CLASE', 'OJO', 'USER', 'NEW_CLASS'])
y = df['NEW_CLASS']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the first two principal components
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.title('PCA of features (first two principal components)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()