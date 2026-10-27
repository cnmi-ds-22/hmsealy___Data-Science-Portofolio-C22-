# Pass 15


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

#### Feature Selection - ANOVA

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
```

```python
# Split into features and target
X = house_df_target_objects.drop(columns=['SalePrice'])
y = house_df_target_objects['SalePrice']

# One-hot encode: each category becomes its own 0/1 column
# drop_first=True drops one category per feature, so the columns don't add up to 1
X = pd.get_dummies(X, drop_first=True, dtype=int)
X.shape
```

```python
# Repeat CV - Linear Regression with ALL encoded object features
n = 500
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[i] = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Repeat CV RMSE (all features): {results.mean().round(2)}")
print(f"Number of features: {X.shape[1]}")
```

```python
# Score every encoded column with an ANOVA F-value
fvalue_selector = SelectKBest(f_regression, k='all')
fvalue_selector.fit_transform(X, y)

# Plot results, biggest first
anova_results = pd.DataFrame({'ANOVA': fvalue_selector.scores_}, index=X.columns)
anova_results.sort_values(by='ANOVA', ascending=False).plot(kind='bar', figsize=(15, 5))
```

```python
# Keep columns with F-value greater than 10
anova = anova_results.sort_values(by='ANOVA', ascending=False)
keep = anova['ANOVA'][anova['ANOVA'] > 10]
print(keep.index.size)
keep
```

```python
# Are the kept features correlated with each other?
X_anova = X[keep.index]
plt.figure(figsize=(20, 16))
sns.heatmap(X_anova.corr(), annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Feature-to-Feature Correlation (ANOVA picks)')
plt.show()
```

```python
# Find columns that are exact copies of another column
# .T flips rows and columns, so duplicated() can compare columns
dupes = X_anova.columns[X_anova.T.duplicated()].tolist()
dupes
```

```python
# Drop exact duplicate columns (they repeat another column word for word)
drop_these = dupes
X_anova = X_anova.drop(columns=drop_these)
X_anova.shape
```

## Processing


### Fit & Assess model- Using linear Regression

```python
# break data into training and testing sets
X = X_anova
y = house_df_target_objects['SalePrice']

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
