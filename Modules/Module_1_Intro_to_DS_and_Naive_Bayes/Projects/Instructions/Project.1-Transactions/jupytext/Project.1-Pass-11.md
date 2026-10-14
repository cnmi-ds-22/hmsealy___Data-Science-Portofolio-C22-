# Project 1 – Pass-11



## Problem Definition




Business problem: Predict whether a customer will make a successfull transaction,


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
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.big.csv'
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


```

```python
# look for nulls, display relative frequency
null_sum = transaction_df.isnull().sum()
null_freq = null_sum / transaction_df.shape[0] * 100
null_freq.sort_values()

#Output- target has 82% percent null values
```

### Column data types

```python
# display data types
transaction_df.dtypes.sort_values()
```

### Unique elements

```python
# number of unique elements in every column (works on the whole dataframe)
transaction_df.nunique().sort_values(ascending=False)
```

```python
# actual unique values in one column
transaction_df['target'].unique()
```

```python

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

### Drop rows with no target
  - The target has 83% nulls. Model cant learn from rows with no answer, remove them first


Selected `target` for the target and drop the rows with null values

```python
# Keep only the rows where target is not missing
transaction_df_target=transaction_df.dropna(subset=['target']).copy()
transaction_df_target.head() #this the check the target rows for null values
```

### Backup #2

```python
# make backup
transaction_df_target_backup = transaction_df_target.copy()
```

```python
# restore from back up
transaction_df_target = transaction_df_target_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.

```python
null_sum = transaction_df_target.isnull().sum() #count the null value in each column
null_sum
```

```python
# turn counts into percentage (part divide whole x 100)
null_freq = null_sum / transaction_df_target.shape[0] * 100
null_freq
```

```python
#find the number of columns with 100% nulls
is_full = null_freq == 100 # ask True/False for each column: "is it 100% null?"

full_null = null_freq[is_full] # keep only the columns where the answer is True

print(len(full_null), 'columns are 100% null') #count them

```

```python
# find the number of column with party empty nulls
more_than_zero = null_freq > 0 #ask two True/False questions for each column
less_than_100 = null_freq < 100


is_partial = more_than_zero & less_than_100 # keep columns where BOTH answers are True (& means "and")
partial_null = null_freq[is_partial]

print(len(partial_null), 'columns are partly null')

```

```python
transaction_df_target.isnull().sum().sort_values (ascending=False) # show me where the null value are
```

```python
# Remove all the null row under target
transaction_df_target_clean =transaction_df_target.dropna(axis ='columns')
transaction_df_target_clean.isnull().sum().sort_values(ascending=False)

transaction_df_target_clean.head()


```

```python
#DROP THE 'UNMAMED:0 and id_code, colums)
transaction_df_target_clean = transaction_df_target_clean.drop (columns=['Unnamed: 0','ID_code'])
transaction_df_target_clean
```

## Recommendations for Data Cleaning

**1. Identifier Columns**
- **Two columns:** `Unnamed: 0` is row numbers and `ID_code` is customer ID. cannot help predict anything
- **Recommendation:** Drop both columns.

**2. Null values:** he `target` column has rows with null values, and some feature columns did too.
- After removing rows with no target, I checked the null percentage of every column: 51 feature columns are 100% null and 0 columns are partly null.
- **Recommendation:** Remove every column that has any missing values.

**3. Target nulls:** 82% null values, rows with nulls are dropped are 100% empty. They have no data to learn from, and nothing to average for imputing.
- **Recommendation:** Drop these rows.

**4. Feature columns with high null counts:** All 51 columns with nulls

**5. Medium null counts:** None. No colum falls between low and high

**6. Low null counts:** None. No column has a small number

**7. Null rows:** After droping the empty columns,no nulls remain



## Exploratory Data Analysis ##


### Normal distribution of features:histogram

```python
# Display a histogram

transaction_df_target_clean.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

```

```python
# Check the distributions of the features

transaction_df_target_clean.hist(
  figsize=(15,15), # size of the whole picture in inches(width,height)
  layout=(10,10),    # 5 rows, 4 column= 20 boxes
  bins=30,         # more bars, so the shape is clear
  edgecolor= "yellow"
)
plt.tight_layout()
plt.show()
#output- each graph look like a bell-shaped (first GNB assumption)
```

### Independence of features: Correlation matrix

```python
# Calculate a correlation matrix (target & features)
transaction_df_target_clean.corr()
```

```python
# Display the correlation matrix

plt.figure(figsize=(8, 6))
sns.heatmap(transaction_df_target_clean.corr(),
            annot=False,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()

```

```python
# select the features
feature_cols =[] #start with an empty list to hold the features
for col in transaction_df_target_clean.columns: #look at each column name
  if col.startswith ('var_'): #if the name start with 'var_, it's a feature
    feature_cols.append(col)

print(feature_cols)
print(len(feature_cols))
```

```python
# Calculate a correlation matrix (features only)
transaction_df_features_only = transaction_df_target_clean[feature_cols]
feature_corr = transaction_df_features_only.corr()

plt.figure(figsize=(25, 25))
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
transaction_df_no_success= transaction_df_target_clean[transaction_df_target_clean['target'] == 0].copy()

# successful transaction (target= 1)
transaction_df_successful= transaction_df_target_clean[transaction_df_target_clean['target'] == 1].copy()


print(type(transaction_df_successful))
print(type(transaction_df_no_success))

print(transaction_df_successful.shape)
print(transaction_df_no_success.shape)

```

## Data Processing


### Single Cross-validation

```python
# Split data from into target and features
target =transaction_df_target_clean['target'].copy()
features= transaction_df_target_clean [feature_cols].copy()


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

```

### Communication of Results

The target show no transaction (‘0’) or  successful transaction (‘1’).

**Accuracy:** 91%


**Confusion matrix:**

 - 39,943: guessed no transaction, actually no transaction → right
  
 - 3513: guessed no transaction, actually successfull transaction → wrong

 - 519: guessed sucessful transaction, actually no transaction -> wrong

 - 1,025: guess sucessful transaction, actually transaction ->right
  
**Classification report:**
  
  - Recall for successfull transaction: 0.23
    - The model found 1025 out 4538 successfull transaction.

  - Precision for successfull transaction: 0.66
    - The model said successfull transaction 1544 times, right 1025 times

  - 90% is just for the no transaction customer.

**Takeaway:** The 91% accuracy mostly comes from the no-transaction customers. The model guesses "no transaction" for most customers and misses about 3 out of 4 buyers.



### Multiple Cross Validation

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

**Mean accuracy:** 91.13% across 100 runs. This is almost the same as the Single CV (91.04%). So the single split was a typical result,not luck

```python
# Display a histogram of the results
plt.hist(results) ;
```

**Histogram observation:** The accuracy scores are tightly grouped between about 90.9% and 91.5%. The shape is roughly bell-shaped. The narrow spread means the model gives about the same accuracy no matter how the data is split.

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

## Communication of Multiple CV Results

**Classification report** (last of the 100 runs):
- **Recall for successful transaction: 0.22**
  - The model found 1,009 out of 4,535 successful transactions.
- **Precision for successful transaction: 0.65**
  - The model said "successful transaction" 1,551 times and was right 1,009 times.
- No-transaction scores high (precision 0.92, recall 0.99), so the 91% accuracy comes mostly from those customers.

**Takeaway:** The model guessed "no transaction" for most customers and missed about 4 out of 5 buyers.


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

**Mean accuracy (balanced data):** 75.4% across 100 runs, lower than the 91.1% on the whole data.

```python
# Plot a histogram
plt.hist(balance_results) ;
```

#### Histogram observation

- Most of the accuracy score is between 74%-76.5%, the accuracy score look a little wider than the whole data histogram because the balanced test set is smaller

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

## Communication of Balanced Data Results

**Classification report** (last of the 100 runs):
- **Recall for successful transaction: 0.74**
  - The model found 3314 out of 4,450 successful transactions.
- **Precision for successful transaction: 0.75**
  - The model said successful transaction 4437 times and was right 3314 times.
- Both classes now score about the same (around 0.75), so the macro and weighted averages match.

**Takeaway:** The model now finds 3 out of 4 buyers, instead of 1 out of 4 with the whole data.


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
| Average accuracy (100 runs)| 92% | 75%|
| Recall:  | 0.23 | 0.74|
|Precision: |0.66|0.76


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

