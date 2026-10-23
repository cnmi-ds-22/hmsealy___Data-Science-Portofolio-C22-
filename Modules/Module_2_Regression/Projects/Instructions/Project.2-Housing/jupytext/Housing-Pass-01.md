# Project Title


## Problem Definition



State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.


## Data Collection/Sources


```python
#import from IPython.display import Image
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

```python
# file name
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv'
```

```python
#load data
house_df= pd.read_csv(url,
               usecols= ['Lot Frontage','SalePrice'])




```

```python
# look at first and last few rows
print(house_df.head())
print(house_df.tail())
```

```python
# look at shape and size

print(house_df.shape)
print(house_df.size)

```

```python
# display meta data (info)
house_df.info()
```

```python
# display summary statistics
house_df.describe()


```

```python
# Columns data types
house_df.dtypes.sort_values()

```

```python
# look for how the often a value appears
house_df['SalePrice'].value_counts(dropna=False)

```

```python
# look for nulls in target and features
house_df.isnull().sum().sort_values(ascending=False)
```

```python
# look for nulls in feature in percentage
house_df['Lot Frontage'].isnull().mean() * 100
```




## Data Cleaning



### Backup 1

```python
# make backup
house_df_backup= house_df.copy()
```

```python
# restore from backup
house_df = house_df_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.


### Impute

```python
# Calculate the median
median_frontage = house_df['Lot Frontage'].median()
print(median_frontage)

# Fill nulls with the median
house_df['Lot Frontage']= house_df['Lot Frontage'].fillna(median_frontage)

#verify there are no nulls
house_df.isnull().sum()

```

```python
house_df['Lot Frontage'].median()
```

```python

```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other

```python
# Calculate a correlation matrix for target
house_df.corr() ;
```

```python
# Display the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(house_df.corr(),
            annot=True,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()
```

```python

```

```python

```

## Processing



```python

```

## Data Visualization/Communication of Results


```python

```
