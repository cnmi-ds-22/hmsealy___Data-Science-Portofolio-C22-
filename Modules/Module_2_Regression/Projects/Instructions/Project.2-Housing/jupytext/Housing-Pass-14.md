# Pass 14


## Problem Definition



Business problem: estimate a house’s sale price.

Data science problem: supervised regression, since we know the real prices and the target is a number.


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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
```

```python
# file name
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv'
```

```python
#load all the data
house_df= pd.read_csv(url)

```

```python
# find all text columns (object)
obj_columns = house_df.select_dtypes(include=['object']).columns.tolist()
obj_columns
```

```python
# this now includes object AND "SalePrice" columns
house_df_target_objects = house_df[obj_columns + ['SalePrice']].copy()
```

```python
# Columns data types
house_df_target_objects.dtypes.sort_values()

```

```python
# look at first and last few rows
house_df_target_objects.head()

```

```python
# look at the last few rows
house_df_target_objects.tail()
```

```python
# look at shape and size

print(house_df_target_objects.shape)
print(house_df_target_objects.size)

```

```python
# display meta data (info)
house_df_target_objects.info()
```

```python
# display summary statistics
house_df_target_objects.describe(include='object')


```

```python
# look for how the often a value appears
house_df['SalePrice'].value_counts(dropna=False)

```

```python
# look for nulls in target and features
house_df_target_objects.isnull().sum().sort_values(ascending=False)
```




## Data Cleaning



### Backup 1

```python
# make backup
house_df_target_objects_backup = house_df_target_objects.copy()
```

```python
# restore from backup
house_df_target_objects = house_df_target_objects_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.

```python
# look for nulls in feature in percentage
house_df_target_objects.isnull().mean() * 100
```

### Impute

```python
# For text columns, a null usually means "this house doesn't have it"
# (no pool, no alley, no garage). So fill with the word 'None'.
house_df_target_objects = house_df_target_objects.fillna('None')

```

```python
#check that null are removed from object
house_df_target_objects.isnull().sum()
```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other

```python
# How many categories does each column have?
house_df_target_objects.nunique().sort_values()
```

```python
# Average price for each category in one feature
house_df_target_objects.groupby('Neighborhood')['SalePrice'].mean().sort_values()
```

```python
# Box plot: price spread for each category
# Boxes at very different heights = the feature is linked to price
# Boxes all at the same height = the feature doesn't tell you much
plt.figure(figsize=(15, 6))
sns.boxplot(data=house_df_target_objects, x='Neighborhood', y='SalePrice')
plt.xticks(rotation=90)
plt.show()
```

```python
# Same box plot for every text feature
for col in obj_columns:
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=house_df_target_objects, x=col, y='SalePrice')
    plt.xticks(rotation=90)
    plt.title(col)
    plt.show()
```
