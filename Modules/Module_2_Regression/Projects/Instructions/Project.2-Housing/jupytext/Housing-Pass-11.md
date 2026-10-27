# Pass 11


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
# find all number columns (int and float)
num_columns = house_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_columns
```

```python
# Drop ID columns: they're labels, not real measurements
# errors='ignore' means no error if a column isn't there
id_columns = ['Order', 'PID']
```

```python
house_df_target_floats = house_df[num_columns].drop(columns=id_columns, errors='ignore').copy()
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
plt.figure(figsize=(20, 16))
sns.heatmap(house_df_target_floats.corr(),
            annot=False,
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
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
plt.figure(figsize=(20, 16))
sns.heatmap(features_corr,
            annot=False,
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
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
# Keep features with correlation above 0.5 (and drop SalePrice itself)
keep = corrs[(corrs > 0.5) & (corrs < 1)]
selected = keep.index.tolist()
selected
```

```python
# Check if the selected features are correlated with each other
plt.figure(figsize=(10, 8))
sns.heatmap(house_df_target_floats[selected].corr(),
            annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Selected Features - Correlation with Each Other')
plt.show()
```

```python
# Drop the lower-scoring feature from each overlapping pair
drop_these = ['Garage Area']
X_selected = house_df_target_floats[selected].drop(columns=drop_these)
X_selected.columns
```

## Processing




### Fit & Assess model- Using linear Regression

```python
# break data into training and testing sets
X = X_selected
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


```

## Data Visualization/Communication of Results


```python

```
