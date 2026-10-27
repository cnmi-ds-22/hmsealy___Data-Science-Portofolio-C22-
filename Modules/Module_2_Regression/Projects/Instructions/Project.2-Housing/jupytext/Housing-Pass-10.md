# Pass 10


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

## Processing




### Fit & Assess model- Using Lasso Regression

```python
# break data into training and testing sets
X = house_df_target_floats.drop(columns=['SalePrice'])
y = house_df_target_floats['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

```python
# Repeat CV - Lasso with ALL float features
n = 500
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Lasso(alpha=1, tol=0.01, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[i] = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Repeat CV RMSE (all features): {results.mean().round(2)}")
print(f"Number of features: {X.shape[1]}")
```

```python
# Try different alphas. Bigger alpha = stronger penalty = more features dropped.
alphas = [1, 10, 50, 100, 200, 500, 1000, 2000]
rmse_list = []
kept_list = []

for a in alphas:
    results = np.zeros(200)
    for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = Lasso(alpha=a, tol=0.01, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[i] = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(results.mean())
    kept_list.append((model.coef_ != 0).sum())

alpha_df = pd.DataFrame({'Alpha': alphas, 'RMSE': rmse_list, 'Features kept': kept_list})
alpha_df
```

```python
# Pick the biggest alpha whose RMSE is within 1% of the best.
# Bigger alpha = fewer features, with almost no loss in accuracy.
best_rmse = alpha_df['RMSE'].min()
close_enough = alpha_df[alpha_df['RMSE'] <= best_rmse * 1.01]
best_alpha = close_enough['Alpha'].max()
best_alpha
```

```python
# Fit Lasso once on scaled data to see each feature's coefficient
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
model = Lasso(alpha=best_alpha, tol=0.01, max_iter=10000)
model.fit(X_train_s, y_train)

coef_df = pd.DataFrame({'Feature': X.columns, 'Lasso': model.coef_})
coef_df['Abs'] = coef_df['Lasso'].abs()
coef_df.sort_values('Abs', ascending=False)
```

```python
# Bar graph of feature importance
coef_df.sort_values('Abs', ascending=False).plot(x='Feature', y='Abs', kind='bar', figsize=(10,5))
plt.ylabel('Lasso coefficient size');
```

```python
# Lasso drops weak features by setting them to 0. Keep the rest.
selected = coef_df[coef_df['Lasso'] != 0]['Feature'].tolist()
selected
```

```python
# Repeat CV - Lasso with SELECTED features
X_sel = X[selected]
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.25)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Lasso(alpha=best_alpha, tol=0.01, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[i] = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Repeat CV RMSE (selected features): {results.mean().round(2)}")
print(f"Number of features: {len(selected)}")
```

```python
# Final fit on selected features, fixed split
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=4)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Lasso(alpha=best_alpha, tol=0.01, max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
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

# (does one miss tell you anything about the next miss? It shouldn't.)
# Across = house's row number (0, 1, 2...). Up = miss. Blue line connects them in order.
# Good = random zigzag around 0. Bad = waves, trends, or long runs on one side.
# My result: random zigzag, no pattern -> OK.
# One big outlier near -$575k: the model guessed one house far too high.
# Rows were shuffled, so this check can't really fail here.

```

## Data Visualization/Communication of Results


```python

```
