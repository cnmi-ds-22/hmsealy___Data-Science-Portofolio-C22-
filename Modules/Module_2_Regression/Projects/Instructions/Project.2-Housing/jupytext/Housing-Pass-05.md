# Pass 05


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
from sklearn.linear_model import Ridge #ridge regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
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

## Processing




### Fit & Assess our model


#### Using Ridge Regression

```python
# break data into training and testing sets
X = house_df[['Lot Frontage']]
y = house_df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

```python
# Fit a Ridge model using sklearn
model = Ridge(alpha=1)      # alpha = penalty strength (1 is the default)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

(model.intercept_, model.coef_)
```

```python
# See how well the model does compared to the test data
plt.scatter(y_test, y_pred)
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], c='red')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price');
```

```python
# RMSE on test data
rmse = mean_squared_error(y_test, y_pred)**0.5
print(f"RMSE: {rmse}")
```

```python
# Use cross validation to assess model performance
results = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=10)
cv_rmse = abs(results.mean())
print(f"CV RMSE: {cv_rmse}")
```

### Residual Analysis

```python
# QQ Plot -checks if residuals (misses) are normally distributed (bell-shaped)
res = y_test - y_pred
sm.qqplot(res, fit=True, line="45") ;
```

```python
# Residuals vs Fitted (Predicted) Values - Constant variance

plt.figure(figsize = (10,6))
plt.scatter(y_pred, res)
plt.xlabel("Fitted (Guess)")
plt.ylabel("Residuals (Miss)")
plt.hlines(0, min(y_pred), max(y_pred), colors = 'red', linestyles = 'dashed') ;
```

```python
# Residuals in Row Order - checks independence
plt.figure(figsize = (10,6))
plt.scatter(range(len(res)), res)
plt.plot(range(len(res)), res, 'b')
plt.xlabel("Test House Order (row number)")
plt.ylabel("Residuals (Miss)")
plt.title("Residuals in Row Order - Independence Check")
plt.hlines(0, min(range(len(res))), max(range(len(res))), colors = 'red', linestyles = 'dashed') ;
```

```python

```

## Communication of Results



**Model:** Ridge Regression, one feature (Lot Frontage)

**Scores:**
- RMSE: $75,253
- CV RMSE: $73,679
- Same as Linear Regression in Pass 4.

**Residuals:**
- QQ plot: skewed right. Expensive houses are guessed too low.
- Fitted plot: no funnel, so roughly OK.
- Row order: random, OK.

**Conclusion:**
Ridge did not improve the model. With one feature, Ridge and Linear give the same answer. Lot frontage alone is a weak predictor. More features are needed.
