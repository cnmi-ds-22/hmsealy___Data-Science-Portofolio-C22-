# Project 1 – Pass-06




## Problem Definition




Business problem: Predict whether a customer will make a successful transaction,


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

#output (# of rows, # of columns)
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


Selected `target` for the target and `var_0 through var_49` for the features

```python
# select the target (All rows= 180,000)
transaction_df_target=transaction_df[['target']].head(180000).copy()
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
#feature_cols= feature_cols #using all the column dont need this code line

#keep all rows 180,000=(.head(# of rows))
transaction_df_target_feature = transaction_df[['target'] + feature_cols].head(180000).copy()

transaction_df_target_feature.shape #verify

```

## Recommendations for Data Cleaning

**1. Identifier Columns**
- **Two columns:** `Unnamed: 0` is row numbers and `ID_code` is customer ID. cannot help predict anything
- **Recommendation:** Drop both columns. (I kept only `target` and `var_0` to `var_49`.)

**2. Null values:** No feature column have nulls. no action needed

**3. Target nulls:** none, no rows dropped

**4. Feature columns with high null counts:** None

**5. Medium null counts:** none

**6. Low null counts:** none

**7. Null rows:** none



## Exploratory Data Analysis ##


### Normal distribution of features:histogram

```python
# Display a histogram - overview of target and the all features

transaction_df_target_feature.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

```

```python
# Check the distributions of the all features (var_0 to var_49)

transaction_df_target_feature[feature_cols].hist(
  figsize=(15,10), # size of the whole picture in inches(width,height)
  layout=(7,8),
  bins=30,         # more bars, so the shape is clear
  edgecolor= "red"
)
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

plt.figure(figsize=(20, 20))
sns.heatmap(transaction_df_target_feature.corr(),
            annot=False,      # change to false, to many numbers
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

plt.figure(figsize=(20, 12))
sns.heatmap(feature_corr,
            annot=False,
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
mat = metrics.confusion_matrix(y_test,pred_result,)
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
print(metrics.classification_report(y_test, pred_result,))



```

### Communication of Single Cross Classification report

The target show no transaction (‘0’) or  successful transaction (‘1’).

**Accuracy:** 91%


**Confusion matrix:**

 - 39,915: guessed no transaction, actually no transaction → right
  
 - 3576: guessed no transaction, actually successful transaction → wrong

 - 511: guessed successful transaction, actually no transaction -> wrong

 - 998: guess successful transaction, actually transaction ->right
  
**Classification report:**
  
  - Recall for successful transaction: 0.22
    - The model found 998 out 4574 successful transaction.

  - Precision for successful transaction: 0.66
    - The model said successful transaction 1509 times, right 998 times   


**Takeaway:** The model guessed "no transaction" for most customer transaction.



## Multiple Cross Validation

```python
# Multiple cross-validation loop
n= 100
results= np.zeros(n)

for i in range(n):
  x_train, x_test, y_train, y_test= train_test_split(features,target,test_size=0.25, random_state=i)
  modelGNB = GaussianNB()
  modelGNB.fit(x_train,y_train)
  y_pred = modelGNB.predict(x_test)
  results[i]=metrics.accuracy_score(y_test,y_pred)

results




```

```python
# Calculate mean of results
results.mean()
```

#### Difference between Single CV accuracy & Multiple CV accuracy

- Single CV accuracy= 90.91%
- Multiple CV accuracy= 91.11%
- Doing 100 run is almost the same as doing 1 run

```python
# Display a histogram of the results
plt.hist(results) ;
```

#### Histogram observation

- Most of the accuracy score is between 90%-91%, the accuracy score barely changed no matter how the data split

```python
# display a confusion matrix

mat= metrics.confusion_matrix(y_test,y_pred,)
sns.heatmap( mat.T,
    square=True,
    annot=True,
    fmt = "d",
    cbar = False,
    xticklabels=names,  #names start left to right
    yticklabels=names,  #names start from top to bottom
)
plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)') ;


```

```python
# Classification report
print(metrics.classification_report(y_test, y_pred))
```

### Communication of Multiple Cross Classification report

**Classification report:**
  
  - Recall for successful transaction: 0.23
    - model caught 1 in 4 real transaction

  - Precision for successful transaction: 0.69
    - when the model guess"transaction it was right about 7 out of 10 times


**Takeaway:** The model guessed "no transaction" for most customer transaction.

## Balanced data set

```python
# Count the transactions databases
number_of_success= len(transaction_df_successful) #len()counts how many rows
print('successful rows:',number_of_success)

number_of_not_successful=len(transaction_df_no_success)
print('No Success rows:',number_of_not_successful)


```

```python
# Create sample of no success row to equal successful rows
transaction_df_no_success_sampled = transaction_df_no_success.sample(n=number_of_success, random_state=42)
print(len(transaction_df_no_success_sampled))
```

```python
# Put the two groups together into one table
balance_transaction_df= pd.concat([transaction_df_successful,transaction_df_no_success_sampled])
```

```python
# Verify target counts
balance_transaction_df['target'].value_counts()
```

```python
#split the balance table into target and feature
balance_target = balance_transaction_df['target'].copy()
balance_features = balance_transaction_df[feature_cols].copy()
```

```python
# Multiple cross-validation loop
n= 100
balance_results= np.zeros(n)

for i in range(n):
  x_train, x_test, y_train, y_test= train_test_split(balance_features,balance_target,test_size=0.25, random_state=i)
  modelGNB = GaussianNB()
  modelGNB.fit(x_train,y_train)
  y_pred = modelGNB.predict(x_test)
  balance_results[i]=metrics.accuracy_score(y_test,y_pred)

balance_results
```

```python
# Calculate mean of balance results
balance_results.mean()
```

```python
# Plot a histogram
plt.hist(balance_results) ;
```

#### Histogram observation

- Most of the accuracy score is between 75%-76.5%, the accuracy score look a little wider than the whole data histogram

```python
# display a confusion matrix

mat= metrics.confusion_matrix(y_test,y_pred,)
sns.heatmap( mat.T,
    square=True,
    annot=True,
    fmt = "d",
    cbar = False,
    xticklabels=names,  #names start left to right
    yticklabels=names,  #names start from top to bottom
)
plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)') ;
```

```python
# Classification report
print(metrics.classification_report(y_test, y_pred))
```

Precision and recall are about 75%. This model can guess 3 out of 4 success transaction instead of 1 out of 4 sucess in the whole data model.


## Data Visualization

```python
# Model 1: Whole data
x_train, x_test, y_train, y_test_whole = train_test_split(features, target, test_size=0.25, random_state=42)
model_whole = GaussianNB()
model_whole.fit(x_train, y_train)
y_pred_whole = model_whole.predict(x_test)


#Model 2: Balance data [cell 63]
x_train, x_test, y_train, y_test_balance= train_test_split(balance_features,balance_target,test_size=0.25, random_state=42)
modelGNB = GaussianNB()
modelGNB.fit(x_train,y_train)
y_pred_balance = modelGNB.predict(x_test)

```

```python
print(results.mean())
print(balance_results.mean())
```

```python
# Display a confusion matrix (Whole data)

mat = metrics.confusion_matrix(y_test_whole, y_pred_whole)
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=names,
            yticklabels=names)

plt.title('Whole data')
plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)')
plt.show()
```

```python
# Display a confusion matrix (Balance data)

mat = metrics.confusion_matrix(y_test_balance, y_pred_balance)
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=names,
            yticklabels=names)

plt.title('Balance data')
plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)')
plt.show()
```

```python
#Classification report (whole data )
print(metrics.classification_report(y_test_whole, y_pred_whole,))
```

```python
#Classification report (Balance data )
print(metrics.classification_report(y_test_balance, y_pred_balance,))
```

# Communication of Results

|   | Whole data | Balanced data|
|---|---|---|
| Average accuracy (100 runs)| 91% | 76%|
| Recall:  | 0.22 | 0.75|
|Precision: |0.66|0.77


### whole data
  - had a higher accuracy percentage because it guess no transaction majority of the time and
  - lower recall because it did not catch customer who made a transaction
  - lower precision it did not guess right transaction made

### Balanced data
  - had a lower accuracy percentage because it guess a higher amount of success transaction
  - higher recall because it did catch customer who made a transaction
  - higher precision it guess right the transaction made

### Conclusion
  - use the balance data because it solve the business problem of predict when a transaction will be made successful. The other model fail at locating successful transaction.

