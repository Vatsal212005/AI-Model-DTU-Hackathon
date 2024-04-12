## Data Maipulation Libraries
import numpy as np
import pandas as pd

## Data Visualisation Libraray
import matplotlib.pyplot as plt
%matplotlib inline
import pylab
import seaborn as sns

## Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC

## Importing essential libraries to check the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

## Warnings
import warnings
warnings.filterwarnings('ignore')



# import drive
from google.colab import drive
drive.mount('/content/drive')



# Load Dataset
# Load Dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/module-4 ML/Cardiovascular risk prediction/data_cardiovascular_risk.csv')
# Dataset First Look
df.head()
df.tail()

# Dataset Rows & Columns count
df.shape


# Dataset Info
df.info()


# Dataset Duplicate Value Count
len(df[df.duplicated()])
# Missing Values/Null Values Count
df.isnull().sum()

# Dataset Columns
df.columns# Dataset Describe
df.describe(include="all")
# Check Unique Values for each variable.
for i in df.columns.tolist():
  print("No. of unique values in ", i , "is" , df[i].nunique(), ".")
# Separating the categorical and continous variable and storing them
categorical_variable=[]
continous_variable=[]

for i in df.columns:
  if i == 'id':
    pass
  elif df[i].nunique() <5:
    categorical_variable.append(i)
  elif df[i].nunique() >= 5:
    continous_variable.append(i)

print(categorical_variable)
print(continous_variable)
# Summing null values
print('Missing Data Count')
df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False)
print('Missing Data Percentage')
print(round(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False)/len(df)*100,2))
# storing the column that contains null values
null_column_list= ['cigsPerDay','BMI','heartRate']
# plotting box plot
plt.figure(figsize=(10,8))
df[null_column_list].boxplot()
# Iterate over the null column list and plot each column's distribution
colors = sns.color_palette("rocket", len(null_column_list))


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))


axes = axes.flatten()


for i, column in enumerate(null_column_list):
    ax = axes[i]

    sns.distplot(df[column], ax=ax, color=colors[i])
    ax.set_title(column)

for j in range(len(null_column_list), len(axes)):
    axes[j].remove()

plt.show()
# Correlation Heatmap visualization code


# Chart - 1 visualization code
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(x="sex", y="age", hue="TenYearCHD", data= df, ax=ax)
ax.set_title("Age Distribution of Patients by Sex and CHD Risk Level")
ax.set_xlabel("Sex")
ax.set_ylabel("Age")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["No Risk", "At Risk"], loc="best")
plt.show()


# Chart - 2 visualization code

plt.figure(figsize=(10,8))
sns.countplot(x='sex', hue='TenYearCHD', data= df)
plt.title('Frequency of CHD cases by gender')
plt.legend(['No Risk', 'At Risk'])
plt.show()

# Chart - 3 visualization code
plt.figure(figsize=(10,8))
sns.countplot(x='is_smoking', hue='TenYearCHD', data= df)
plt.title('A Comparison of Smokers and Non-Smokers')
plt.legend(['No Risk', 'At Risk'])
plt.show()
#Chart - 4 visualization code
plt.figure(figsize=(10,8))
sns.countplot(x= df['cigsPerDay'],hue= df['TenYearCHD'])
plt.title('How much smoking affect CHD?')
plt.legend(['No Risk','At Risk'])
plt.show()
# Chart - 8 visualization code
plt.figure(figsize=(10,8))
sns.barplot(x=df['diabetes'], y=df['TenYearCHD'], hue=df['TenYearCHD'], estimator=lambda x: len(x) / len(df) * 100)
plt.title('Proportion of patients with and without diabetes at CHD risk')
plt.xlabel('Diabetes')
plt.ylabel('Percentage')
plt.legend(title='CHD Risk', labels=['No Risk', 'At Risk'])
plt.show()


# Chart - 10 visualization code
cols = ['glucose','TenYearCHD']

# create the scatter plot matrix
plt.figure(figsize=(15,10))
sns.pairplot(df[cols], hue='TenYearCHD', markers=['o', 's'])
plt.show()
# Chart - 11 visualization code
plt.figure(figsize=(10,8))
cols = ['sex', 'cigsPerDay', 'TenYearCHD']
sns.scatterplot(x='cigsPerDay', y='TenYearCHD', hue='sex', data=df)
plt.show()
# Correlation Heatmap visualization code
# Drop non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Plot correlation heatmap
plt.figure(figsize=(12, 12))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap="mako")
plt.show()