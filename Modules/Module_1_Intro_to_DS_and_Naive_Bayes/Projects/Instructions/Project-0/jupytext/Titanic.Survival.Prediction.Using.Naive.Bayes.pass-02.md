# Naive Bayes Classification to Predict Titanic Survival


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition

State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.


## Data Collection/Sources


### Data Overview


This data comes from [Kaggle](https://www.kaggle.com/c/titanic/data), which was then copied to AWS for easy access.

https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv



### Imports

```python
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn import datasets, metrics, model_selection
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
```

### File Paths

Create a variable `url` that contains the web link to the CSV data

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv'  # insert url here
url
```

### Load data

Load the data into a data frame named `titanic_df`.

```python
# load data
titanic_df = pd.read_csv( url )
```


### head


Look at the first few rows

```python
# first few rows
titanic_df.head()
```


### tail


Look at the last few rows


```python
# last few rows
titanic_df.tail()
```


### shape


```python
# shape of data frame
titanic_df.shape  #output: display (row,column)
```

### size


```python
# size of data frame
titanic_df.size   #output: calculate the number of rows multiplied by the nuber of columns
```


### Summary meta-data


Run the `info()` method on the data frame.


```python
# display meta-data
titanic_df.info()

# display show:
#total rows= 891
#total colun=12
#non-null count= show which columns have missing data by reporting( anything less than 891)
#data type= int(whole number),float(decimal number), object(text string or mixed data)

```

### Summary statistics

Summary statistics give you a quick overview of the data.  Some items to spot check:
- the Target:
  - look for nulls
    - null will be culled
  - look at min, max, median, and mean
    - is it a categorical field masquerading as a number?
- the Features:
  - numerical
    - look at min, max, median, and mean
      - a min=1 and max=count suggests an ID
  - categorical
    - look at unique and freq
      - freq=1, unique=count suggests an ID
      - freq=count suggests a meaningless feature


Run the `describe()` method on the data frame.

```python
# display summary statistics
titanic_df.describe()
```


### target value_counts()



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# value_counts on target ; include nulls
titanic_df['Survived'].value_counts( dropna = False )

#value_counts, count all the row but skip blank rows
#dropna = false, blank rows are counted

#output shows
#0-> passenger who died
#1-> passenger who survived, #1 usually mean yes
# since the output onluy have 2 valvues-> this is classigication
```


```python
# normalized value_counts on target ; include nulls
titanic_df['Survived'].value_counts( normalize = True , dropna = False ) * 100

# normalize = True, take the count and divided by the total amount of rows
# ex. 549/ 891 & 342/891.

```

### Nulls



```python
# look for nulls
titanic_df.isnull().sum()

#.isnull, this look at the table and ask if the cell blank, true of false
#.sum(), add up the column with the blanks(True)
```


```python
# look for nulls, displaying relative frequency
null_sum = titanic_df.isnull().sum()
null_freq = null_sum / titanic_df.shape[0] *100
#.shape give (rows,columns)but [0]pick the first position(rows)
# the sum of all the null values/ number for rows
null_freq.sort_values()
#

```

### dtypes - column data types


Which data types would be appropriate for Gaussian Naive Bayes?


```python
# display data types
titanic_df.info() # float and interger work best, can do math
                  # cannot use object cuz this is text. can't no math
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
titanic_df.nunique(axis="rows")

# ().nunique)counts diffrent values in the column,
# (axis="rows") means it counts down each colum
# unique item, ask can this information show a pattern

# If the count = number of rows (891), every person has their own value.
# That's an ID, like PassengerId or Name. It can't show a pattern, so drop it.

# Many values isn't always bad: Age and Fare are amounts, so keep them.
# Always ask: can this column show a pattern?
```




## Data Cleaning



### Backup #1


```python
# make backup
backup_titanic_df=titanic_df.copy() # .copy() makes a separate backup before cleaning
backup_titanic_df                 # this variable has the orginal 12 columns
```

```python
# restore from backup
titanic_df = backup_titanic_df # this restore
```


### Keep only the target and one feature


Select `Survived` for the target and `Age` for the single feature

```python
# select the target and one feature
titanic_df =titanic_df[['Survived','Age']] # this change this variable to hold 2 columns
titanic_df
```

### Backup #2


```python
# make backup
data_cleaned_titanic_df= titanic_df.copy()  #this save the 2 column in to a new variable
data_cleaned_titanic_df # this the name of the variable with 2 columns
```

```python
# restore from backup=
titanic_df= data_cleaned_titanic_df.copy()  # this restore back to the 2 columns

```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.



#### Impute


Impute with the median.


```python
# Calculate the median
median_age =data_cleaned_titanic_df['Age'].median()  # this find the middle age for all passengers
median_age  #middle value, line everyone up from youngest to olderet the person in the middle
```

```python
# Assign median to nulls
data_cleaned_titanic_df['Age'].fillna(median_age, inplace=True) # this fill any blank row under age with 28(median age)
```

```python
# Verify there are no nulls
data_cleaned_titanic_df.isnull().sum()  #this count blank cell in the column(rows)
```

```python
# Verify median is about the same as before
data_cleaned_titanic_df.median()
```

## Exploratory Data Analysis



GNB has two requirements:
1. the features are normally distributed
1. the features are not correlated with each other






### normal distribution of features: histogram


```python
# Display a histogram
data_cleaned_titanic_df['Age'].hist(). # output, bar chart showing how many passengers fall into each age range
```


### no correlation with single feature




## Processing



```python
# Split data frame into target and features
target= data_cleaned_titanic_df[['Survived']]
features = data_cleaned_titanic_df[['Age']]
```

```python
# Verify features
features
```

```python
# Verify target
target
```

### Single Cross Validation ( CV )


Run a single Cross Validation


#### Train Test Split

```python
# Train Test Split
x_train, x_test, y_train, y_test= train_test_split(features,target, test_size=0.25)

#train_test_split only work if you import (from sklearn.model_selection import train_test_split )

# x=features(clues)
# y= target(answers)

# x_train= age for the training group
# x_test= age of the test group
# y_train= survived answer for the training group
# y_test= survived answers for the test group
```

```python
# Verify shape of each data set from Train Test Split
for i in (x_train, x_test, y_train, y_test):
  print(i.shape)

#668 + 223 = 891. Nobody got lost. ✓
#223 ÷ 891 ≈ 25%, matching test_size=0.25. ✓

```

#### Model


```python
# Create an empty GNB model
modelGNB = GaussianNB()
```

#### Fit



```python
# Train ( fit ) the model
modelGNB.fit(pd.DataFrame (x_train), y_train)

#.fit(...) → “learn from this data.” This is the training step.
#x_train → the ages of the 668 training passengers.
#y_train → whether each of them survived.
```

#### Prediction


```python
# Make predicitons using the model
pred_results =modelGNB.predict(pd.DataFrame(x_test))
pred_results

#.predict(...) → “make guesses.” The model looks at each age and decides 0 or 1.
#x_test → the 223 ages it has never seen.
```

#### Performance



We'll meassure accuracy, which is the proportions of the number correctly predicted, which is mathematically the same as one minus the proportion incorrectly predicted.

```python
# Calculate the accuracy
accuracy =metrics.accuracy_score(y_test,pred_results)
accuracy*100

#.accuracy_score(y_test, pred_results) → compares the real answers with the model’s guesses, one passenger at a time, and returns the share it got right
```

```python
# Display a confusion matrix

# EXTRA: chart size (turned off)
#plt.figure(figsize=(8,8))

# NEEDED: count right and wrong guesses
mat = metrics.confusion_matrix(y_test, pred_results)

# NEEDED: draw the grid
sns.heatmap(mat.T,      # EXTRA: .T flips the grid
    square=True,        # EXTRA: square boxes
    annot=True,         # NEEDED: show numbers
    fmt='d',            # NEEDED: whole numbers
    cbar=False,         # EXTRA: hide color bar

)

# EXTRA: label the sides
plt.xlabel('Actual')
plt.ylabel('Predicted')


#output	R
#Right guesses
#138 → The model said “died.” They did die. ✓
#6 → The model said “lived.” They did live. ✓

#Wrong guesses
#75 → The model said “died.” But they lived. ✗
#4 → The model said “lived.” But they died. ✗
```

### Multiple Cross Validation ( CV )


Create a CV loop with 100 iterations.  Capture performance metric in an array names `results`.
Use accuracy for the performance metric.


```python
# Multiple CV loop
results =[]  #make an empty list for the score
for i in range(100): #repeat 100 times
  x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25) #new  random split each time
  modelGNB = GaussianNB() #new empty model
  modelGNB.fit(pd.DataFrame (x_train), y_train) # train it
  pred_results =modelGNB.predict(pd.DataFrame(x_test)) #predicts
  accuracy = metrics.accuracy_score(y_test, pred_results) #score it
  results.append(accuracy) #save the score
results = np.array(results)
results #show all 100 scores



```

```python
# Calculate mean of results
results.mean()
```

```python
# Display a confusion matrix
mat = metrics.confusion_matrix(y_test, pred_results)
sns.heatmap(mat.T,      # EXTRA: .T flips the grid
    square=True,        # EXTRA: square boxes
    annot=True,         # NEEDED: show numbers
    fmt='d',            # NEEDED: whole numbers
    cbar=False,
)
```

```python
# Display a histogram of the results
plt.hist(results)
```


## Communication of Results


Conclusion:


Future work:


