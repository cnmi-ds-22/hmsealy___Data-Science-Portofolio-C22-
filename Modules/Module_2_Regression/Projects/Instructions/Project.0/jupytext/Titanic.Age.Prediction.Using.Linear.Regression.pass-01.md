# Linear Regression to Predict Titanic Passenger Age


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition

State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.


### Regression

For a regression, the Age will be the target.



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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

### File Paths

Create a variable `url` that contains the web link to the CSV data

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv'    # insert url here
url
```

### Load data

Load the data into a data frame named `titanic_df`.

```python
# load data
titanic_df= pd.read_csv(url)
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
titanic_df.shape
```

### size


```python
# size of data frame
titanic_df.size
```


### Summary meta-data


Run the `info()` method on the data frame.


```python
# display meta-data
titanic_df.info()
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


### target nulls



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# look for nulls in the target
titanic_df['Age'].value_counts(dropna=False)
```


### Nulls



```python
# look for nulls
titanic_df.isnull().sum().sort_values(ascending=False)
```


```python
# look for nulls, displaying relative frequency
null_sum= titanic_df.isnull().sum()
null_freq= null_sum/titanic_df.shape[0]*100
null_freq.sort_values()


```

### dtypes - column data types


Which data types would be appropriate for Linear Regression?


```python
# display data types
titanic_df.dtypes.sort_values()
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
titanic_df.nunique().sort_values(ascending=False)
```




## Data Cleaning



### Backup #1


```python
# make backup
titanic_df_backup= titanic_df.copy()
```

```python
# restore from backup
titanic_df = titanic_df_backup.copy()
```


### Keep only the target and one features





Select `Age` for the target and `Fare` for the feature

```python
# select the target and two feature
titanic_df_age_fare = titanic_df[['Age','Fare']]
titanic_df_age_fare.head()
```

### Backup #2


```python
# make backup
titanic_df_age_fare_backup = titanic_df_age_fare.copy()

```

```python
# restore from backup
titanic_df_age_fare_backup = titanic_df_age_fare_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.



#### Impute

```python
# counts the nulls in age and fare
titanic_df_age_fare.isnull().sum().sort_values (ascending=False)
```

```python
# drop the nulls values in Age
titanic_df_age_fare = titanic_df_age_fare.dropna(subset=['Age'])

```

```python
# verify nulls values in Age
titanic_df_age_fare.isnull().sum()
```

Impute with the median.


```python
# Calculate the median

```

```python
# Assign median to nulls

```

```python
# Verify there are no nulls

```

```python
# Verify median is about the same as before

```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other


### correlated with the target


```python
# Calculate a correlation matrix for the target
titanic_df_age_fare.corr() ;
```

```python
# Display the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(titanic_df_age_fare.corr(),
            annot=False,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()
```

```python
# Create a data series with features ranked by correlation

```





### independence of features: correlation

```python
# Calculate a correlation matrix for the features

```

```python
# Display the correlation matrix

```

```python
# Create a data series with features ranked by correlation

```

## Processing



## Residual analysis



## Communication of Results


Conclusion:


Future work:



