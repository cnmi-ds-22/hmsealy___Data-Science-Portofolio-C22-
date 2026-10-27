# Pass 08


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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
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
# find the float column and make it into a
floats_columns = house_df.select_dtypes(include='float64').columns.tolist()
floats_columns
```

```python
target_floats= ['SalePrice'] + floats_columns
```

```python
house_df_target_floats= house_df[target_floats]
```

```python
# Columns data types
house_df_target_floats.dtypes.sort_values()

```

```python
# look at first and last few rows
house_df_target_floats.head()

```

```python
# look at the last few rows
house_df_target_floats.tail()
```

```python
# look at shape and size

print(house_df_target_floats.shape)
print(house_df_target_floats.size)

```

```python
# display meta data (info)
house_df_target_floats.info()
```

```python
# display summary statistics
house_df_target_floats.describe()


```

```python
# look for how the often a value appears
house_df['SalePrice'].value_counts(dropna=False)

```

```python
# look for nulls in target and features
house_df_target_floats.isnull().sum().sort_values(ascending=False)
```




## Data Cleaning



### Backup 1

```python
# make backup
house_df_target_floats_backup = house_df_target_floats.copy()
```

```python
# restore from backup
house_df_target_floats = house_df_target_floats_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.

```python
# look for nulls in feature in percentage
house_df_target_floats.isnull().mean() * 100
```

### Impute

```python
# Calculate the median
median_frontage = house_df_target_floats['Lot Frontage'].median()
median_garage_yr = house_df_target_floats['Garage Yr Blt'].median()

print(median_frontage)
print(median_garage_yr)

# Fill nulls with the median
house_df_target_floats['Lot Frontage']= house_df_target_floats['Lot Frontage'].fillna(median_frontage)
house_df_target_floats['Garage Yr Blt']= house_df_target_floats['Garage Yr Blt'].fillna(median_garage_yr)

#verify there are no nulls
house_df_target_floats.isnull().sum()

```

```python
print(house_df_target_floats['Lot Frontage'].median())
print(house_df_target_floats['Garage Yr Blt'].median())
```

```python
house_df_target_floats.isnull().sum() * 100000
```

```python
# drop the nulls values in features
house_df_target_floats = house_df_target_floats.dropna(subset=['Mas Vnr Area',
'BsmtFin SF 1',
'BsmtFin SF 2',
'Bsmt Unf SF',
'Total Bsmt SF',
'Bsmt Full Bath',
'Garage Cars',
'Garage Area',
'Bsmt Half Bath'])
```

```python
#check that null are removed from features
house_df_target_floats.isnull().sum()
```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other

```python
# Calcualte & display  correlation matrix, check if feature corelated with target
plt.figure(figsize=(8, 6))
sns.heatmap(house_df_target_floats.corr(),
            annot=True,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()
```

```python
# Calculate a correlation matrix for features & features
features_df = house_df_target_floats.drop(columns=['SalePrice'])
features_corr = features_df.corr()
```

```python
# Display the matrix, check if features are correlated with each other
plt.figure(figsize=(12, 10))      # bigger, since there are more columns
sns.heatmap(features_corr,
            annot=True,
            fmt='.2f',            # round numbers to 2 decimals so they fit
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)
plt.title('Feature-to-Feature Correlation Matrix')
plt.show()
```

#### Feature Selection

```python
# Score each feature by how strongly it moves with SalePrice
corrs = house_df_target_floats.corr()['SalePrice'].abs().sort_values(ascending=False)
corrs
```

```python
# Bar graph of the scores (drop SalePrice, since it always scores 1 with itself)
corrs.drop('SalePrice').plot(kind='bar', figsize=(10,5))
plt.ylabel('Correlation with SalePrice (absolute)');
```

```python
# Pick the single strongest feature
best_feature = corrs.drop('SalePrice').index[0]
best_feature
```

## Processing




### Fit & Assess model- Using linear Regression

```python
# break data into training and testing sets
X = house_df_target_floats[[best_feature]]
y = house_df_target_floats['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

```python
# Fit a linear model using sklearn
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

(model.intercept_, model.coef_)
```

```python
# See how well the model does compare to the test data

plt.scatter(y_test, y_pred)
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], c='red')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price');

list(zip(y_test, y_pred))[:10]
```

```python
# RMS on test data
rmse = mean_squared_error(y_test, y_pred)**0.5
print(f"RMSE: {rmse} ")

#RMSE- shows the amount the model was wrong
```

```python
# Use cross validation to assess model performance
results = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=10)
cv_rmse = abs(results.mean())
print(f"CV RMSE: {cv_rmse}")

#Run validation multiple times compare to single cv number barely change
#the single sv lucky that how the model work
```

## Residual Analysis

```python
# QQ Plot - Checks if residuals(misses)are normally distributed (bell-shaped)
# res= real price- guess. Positive= guessed to low. Negative= guesses to high
# fit=True, converts model misses from dollars into standard units
# line="45" draws a line where "should be" = "actually is
# Read it: dots on red line= normal. Ends curving away= not normal
# My result: middle is on the line, right end curves up-> skewed right

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
# Residuals vs Time - checks indepedence
plt.figure(figsize = (10,6))
plt.scatter(range(len(res)), res)
plt.plot(range(len(res)), res, 'b')
plt.xlabel("Test House Order (row number)")
plt.ylabel("Residuals (Miss)")
plt.title("Residuals in Row Order - Independence Check")
plt.hlines(0, min(range(len(res))), max(range(len(res))), colors = 'red', linestyles = 'dashed') ;

# (does one miss tell you anything about the next miss? It shouldn't.)
# Across = house's row number (0, 1, 2...). Up = miss. Blue line connects them in order.
# Good = random zigzag around 0. Bad = waves, trends, or long runs on one side.
# My result: random zigzag, no pattern -> OK.
# Spikes go higher up than down = same right skew as before.
# Caveat: rows were shuffled by train_test_split, so "Time" is just random order.
# This check can't really fail here. Sorting by sale date would make it meaningful.
```

## Data Visualization/Communication of Results


```python

```
