# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
%matplotlib inline
# Load Dataset
df = pd.read_csv("../data/data.csv")
# Basic Overview
print("Shape of dataset:", df.shape)
print("\nData types:\n", df.dtypes)
display(df.head())

# Convert TransactionStartTime to datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
print("\nConverted TransactionStartTime to datetime.")
# Summary Statistics
display(df.describe(include='all'))
# Numerical Distribution
numerical_cols = ['Amount', 'Value']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
# Categorical Feature Distribution
categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 
                    'ChannelId', 'PricingStrategy', 'FraudResult']

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Correlation Matrix

plt.figure(figsize=(6, 4))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Missing Values
print("Missing Values:\n", df.isnull().sum())
msno.matrix(df)
plt.show()
# Outlier Detection
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Time-Based Analysis
df['Hour'] = df['TransactionStartTime'].dt.hour
df['DayOfWeek'] = df['TransactionStartTime'].dt.day_name()

plt.figure(figsize=(8, 4))
sns.countplot(x='Hour', data=df)
plt.title('Transaction Count by Hour')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='DayOfWeek', data=df, order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Transaction Count by Day of Week')
plt.xticks(rotation=45)
plt.show()
# Fraud by Time
plt.figure(figsize=(8, 4))
sns.countplot(x='Hour', hue='FraudResult', data=df)
plt.title('Fraud Count by Hour of Day')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='DayOfWeek', hue='FraudResult', data=df,
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Fraud Count by Day of Week')
plt.show()
# Summary of Key Insights (Markdown Cell in Jupyter recommended)

from IPython.display import Markdown as md
md("""
### 📌 Key EDA Insights

1. **Skewed Features**: `Amount` and `Value` are right-skewed; may need transformation.
2. **Temporal Patterns**: Fraud transactions show clustering at certain hours or days.
3. **Class Imbalance**: `FraudResult` is imbalanced — resampling techniques needed.
4. **Outliers Detected**: High-value outliers visible in `Amount` and `Value`.
5. **Missing Data**: Some nulls in timestamp and possibly other fields to handle.
""")
