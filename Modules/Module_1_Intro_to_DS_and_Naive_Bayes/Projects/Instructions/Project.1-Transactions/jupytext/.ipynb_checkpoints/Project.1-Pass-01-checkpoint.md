# Project 1 – Pass-01




## Problem Definition




Business problem: Predict whether a customer will make a transaction,


Data Science problem: This is a supervised learning and classification problem. I will use the Gaussian Naive Bayes model






## Data Collection


### Imports

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, model_selection
```

### File Paths

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.csv'
```

### Load data

```python
#load data
transaction_df= pd.read_csv(url)
```

### Head

```python
# first few rows
transaction_df.head()
```

### Tail

```python
# last few rows
transaction_df.tail()
```

### Shape

```python
# shape of data frame
transaction_df.shape
```

### Size

```python
# size of data frame
transaction_df.size
```

### Info

```python
# display meta-data
transaction_df.info()

```

## Summary statistics


### Describe

```python
# display summary statistics
transaction_df.describe()

```

### Target value counts


NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.

```python
# value_counts on target ; include nulls
transaction_df['target'].value_counts(dropna=False)
```

```python
#normalize value counts on target ; include nulls
transaction_df['target'].value_counts(normalize=True, dropna=False)
```

### Nulls

```python
# look for nulls
transaction_df.isnull().sum().sort_values(ascending=False)

#output null rows:none (every column has 0 nulls, so no row can contain one)
```

### Column data types

```python
# display data types
transaction_df.dtypes.sort_values()
```

###Unique elements

```python
# number of unique elements in every column (works on the whole dataframe)
transaction_df.nunique().sort_values(ascending=False)
```

```python
# actual unique values in one column
transaction_df['target'].unique()
```

## Data Cleaning




Backup #1

```python
# backup copy of the original data
transaction_df_backup= transaction_df.copy()

```

```python
# restore from backup
transaction_df=transaction_df_backup.copy()
```

### Keep only the target and features


Selected `target` for the target and `var_0 through var_4` for the features

```python
# select the target (first 1000 rows)
transaction_df_target=transaction_df[['target']].head(1000).copy()
transaction_df_target.shape
```

```python
# select the features
feature_cols =[] #start with an empty list to hold the features
for col in transaction_df.columns: #look at each column name
  if col.startswith ('var_'): #if the name start with 'var_, it's a feature
    feature_cols.append(col)
# check the list (show 50 names)
print(feature_cols)
print(len(feature_cols))
```

```python
# select the target and the features
feature_cols= feature_cols[0:5] #keep only the first 5 feature (var0 to var4)

#keep only the first 1000 rows,
transaction_df_target_feature = transaction_df[['target'] + feature_cols].head(1000).copy()

transaction_df_target_feature.shape #verify

```

## Recommendations for Data Cleaning

**1. Identifier Columns**
- **Two columns:** `Unnamed: 0` is row numbers and `ID_code` is customer ID. cannot help predict anything
- **Recommendation:** Drop both columns. (I kept only `target` and `var_0` to `var_4`.)

**2. Null values:** No feature column have nulls. no action needed

**3. Target nulls:** none, no rows dropped

**4. Feature columns with high null counts:** None

**5. Medium null counts:** none

**6. Low null counts:** none

**7. Null rows:** none



## Exploratory Data Analysis ##


### Normal distribution of features:histogram

```python
# Display a histogram - overview of target and the 5 features

transaction_df_target_feature.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

```

```python
# Check the distributions of the 5 features (var_0 to var_4)

transaction_df_target_feature[feature_cols].hist(figsize=(15, 4), bins=30, layout=(1, 5))
plt.tight_layout()
plt.show()

#output- each graph look like a bell-shaped (first GNB assumption)


```

### Independence of features: Correlation matrix

```python
# Calculate a correlation matrix (target & features)
transaction_df_target_feature.corr()
```

```python
# Display the correlation matrix

plt.figure(figsize=(16, 12))
sns.heatmap(transaction_df_target_feature.corr(),
            annot=True,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()

```

```python
# Calculate a correlation matrix (features only)
transaction_df_features_only = transaction_df_target_feature[feature_cols]
feature_corr = transaction_df_features_only.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(feature_corr,
            annot=True,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white/gray = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix - Features Only')
plt.show()

#output- correlation between features are near 0
#look independent, 2nd GNB assumption

```

```python
# Create two data frames

# no transaction (target= 0)
transaction_df_no_success= transaction_df_target_feature[transaction_df_target_feature['target'] == 0].copy()

# successful transaction (target= 1)
transaction_df_successful= transaction_df_target_feature[transaction_df_target_feature['target'] == 1].copy()


print(type(transaction_df_successful))
print(type(transaction_df_no_success))

print(transaction_df_successful.shape)
print(transaction_df_no_success.shape)

```

## Data Processing


### Single Cross-validation

```python
# Split data from into target and features
target =transaction_df_target['target'].copy()
features= transaction_df_target_feature [feature_cols].copy()


```

```python
# Train test split
x_train, x_test, y_train, y_test = train_test_split( features, target, test_size=0.25, random_state=42)
```

```python
# verify shape of each data set from Train test split
for i in (x_train, x_test,y_train,y_test):
  print(i.shape)
  print(type(i))
```

### Model

```python
# Create an empty GNB model
modelGNB= GaussianNB()
```

### Fit

```python
# Train ( fit ) the model
modelGNB.fit(x_train,y_train)
```

### Prediction

```python
# Make prediction using the model
pred_result= modelGNB.predict(x_test)
```

### Performance

```python
# Calculate the accuracy
accuracy=metrics.accuracy_score(y_test,pred_result)
accuracy*100
```

```python
# Display a confusion matrix
mat = metrics.confusion_matrix(y_test,pred_result)
names = ['No transaction', 'Transaction']
sns.heatmap(mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=names,  #names start left to right
    yticklabels=names,  #names start from top to bottom
)

plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)') ;
```

```python
#Classification report
print(metrics.classification_report(y_test, pred_result,zero_division=0 ))

#output show a warning message because model did not guess successful transactio
# zero_division=0 to remove the warning message

```

### Communication of Results

The target show no transaction (‘0’) or  successful transaction (‘1’).

**Accuracy:** 90.4%


**Confusion matrix:**

 - 226: guessed no transaction, actually no transaction → right
  
 - 24: guessed no transaction, actually successfull transaction → wrong
  
**Classification report:**
  
  - Recall for successfull transaction: 0.00
  The model did not find any successfull transaction.
  
  - 90.4% is just for the no transaction customer.

**Takeaway:** The model guessed "no transaction" for every customer.


