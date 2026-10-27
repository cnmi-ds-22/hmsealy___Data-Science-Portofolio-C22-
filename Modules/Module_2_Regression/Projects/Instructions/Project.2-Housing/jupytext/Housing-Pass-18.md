# Pass 18


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
# Keep ALL columns, but drop the ID columns (labels, not measurements)
house_df_all = house_df.drop(columns=['Order', 'PID'], errors='ignore').copy()

# list of number columns (includes SalePrice)
num_columns = house_df_all.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_columns
```

```python
# Columns data types
house_df_all.dtypes.sort_values()

```

```python
# look at first and last few rows
house_df_all.head()

```

```python
# look at the last few rows
house_df_all.tail()
```

```python
# look at shape and size

print(house_df_all.shape)
print(house_df_all.size)

```

```python
# display meta data (info)
house_df_all.info()
```

```python
# summary statistics: numbers first, then text
display(house_df_all.describe())
display(house_df_all.describe(include='object'))


```

```python
# look for how the often a value appears
house_df['SalePrice'].value_counts(dropna=False)

```

```python
# look for nulls in target and features
house_df_all.isnull().sum().sort_values(ascending=False)
```




## Data Cleaning



### Backup 1

```python
# make backup
house_df_all_backup = house_df_all.copy()
```

```python
# restore from backup
house_df_all = house_df_all_backup.copy()
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.

```python
# look for nulls in feature in percentage
house_df_all.isnull().mean() * 100
```

### Impute

```python
# 1. Number columns: fill with the median (same as Pass 13)
median_frontage = house_df_all['Lot Frontage'].median()
median_garage_yr = house_df_all['Garage Yr Blt'].median()
house_df_all['Lot Frontage'] = house_df_all['Lot Frontage'].fillna(median_frontage)
house_df_all['Garage Yr Blt'] = house_df_all['Garage Yr Blt'].fillna(median_garage_yr)

# 2. Number columns with only a few nulls: drop those rows (same as Pass 13)
house_df_all = house_df_all.dropna(subset=['Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
                                           'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath',
                                           'Bsmt Half Bath', 'Garage Cars', 'Garage Area'])

# 3. Text columns: null means "doesn't have it", so fill with 'None'
house_df_all = house_df_all.fillna('None')

house_df_all.shape

```

```python
#check that null are removed from object
house_df_all.isnull().sum()
```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other

```python
# How many categories does each column have?
house_df_all.nunique().sort_values()
```

```python
# Average price for each category in one feature
house_df_all.groupby('Neighborhood')['SalePrice'].mean().sort_values()
```

```python
# Box plot: price spread for each category
# Boxes at very different heights = the feature is linked to price
# Boxes all at the same height = the feature doesn't tell you much
plt.figure(figsize=(15, 6))
sns.boxplot(data=house_df_all, x='Neighborhood', y='SalePrice')
plt.xticks(rotation=90)
plt.show()
```

```python
# Same box plot for every text feature
for col in obj_columns:
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=house_df_all, x=col, y='SalePrice')
    plt.xticks(rotation=90)
    plt.title(col)
    plt.show()
```

```python
# Number features: correlation heatmap (text columns can't be in .corr())
plt.figure(figsize=(20, 16))
sns.heatmap(house_df_all[num_columns].corr(), annot=False,
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix - Number Features')
plt.show()
```

#### Feature Selection

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
```

```python
# Target
y = house_df_all['SalePrice']

# Number features (drop the target)
X_num = house_df_all[num_columns].drop(columns=['SalePrice'])

# Text features, one-hot encoded
X_text = pd.get_dummies(house_df_all[obj_columns], drop_first=True, dtype=int)

# Everything together, for the starting score
X = pd.concat([X_num, X_text], axis=1)

print(X_num.shape, X_text.shape, X.shape)
```

```python
# Find columns that are exact copies of another column
# Twins confuse the model: it can't tell which one deserves the credit
dupes = X.columns[X.T.duplicated()].tolist()
print(dupes)

X = X.drop(columns=dupes)
X.shape
```

## Processing


### Fit & Assess model - Using Lasso Regression

```python
# Repeat CV - Lasso with ALL features (numbers + encoded text)
# alpha=100: a stronger penalty settles much faster than alpha=1 with 268 features
n = 100
results = np.zeros(n)

for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Lasso(alpha=100, tol=0.01, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[i] = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Repeat CV RMSE (all features): {results.mean().round(2)}")
print(f"Number of features: {X.shape[1]}")
```

```python
# Try different alphas. Bigger alpha = stronger penalty = more features set to 0.
alphas = [50, 100, 200, 500, 1000, 2000]
rmse_list = []
kept_list = []

for a in alphas:
    scores = np.zeros(50)
    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = Lasso(alpha=a, tol=0.01, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[i] = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(scores.mean())
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
# Bar graph of the 40 biggest coefficients
coef_df.sort_values('Abs', ascending=False).head(40).plot(x='Feature', y='Abs', kind='bar', figsize=(15,5))
plt.ylabel('Lasso coefficient size');
```

```python
# Lasso drops weak features by setting them to 0. Keep the rest. No cutoff needed.
selected = coef_df[coef_df['Lasso'] != 0]['Feature'].tolist()
print(len(selected))
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
```
