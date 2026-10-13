# Project 1 – Pass-04



## Problem Definition



Business problem: Predict whether a customer will make a transaction,

Data Science problem: This is a supervised learning and classification problem. I will use the Gaussian Naive Bayes model





## Data Collection

### Imports


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, model_selection
```

### File Paths


```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.csv'
```

### Load data


```python
#load data
transaction_df= pd.read_csv(url)
```

### Head


```python
# first few rows
transaction_df.head()
```





  <div id="df-3c7d7a08-aefb-49a9-a841-e9561ba6767d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID_code</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>...</th>
      <th>var_40</th>
      <th>var_41</th>
      <th>var_42</th>
      <th>var_43</th>
      <th>var_44</th>
      <th>var_45</th>
      <th>var_46</th>
      <th>var_47</th>
      <th>var_48</th>
      <th>var_49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>train_0</td>
      <td>0</td>
      <td>8.9255</td>
      <td>-6.7863</td>
      <td>11.9081</td>
      <td>5.1187</td>
      <td>5.7470</td>
      <td>14.0137</td>
      <td>0.5745</td>
      <td>...</td>
      <td>5.4879</td>
      <td>-4.7645</td>
      <td>3.1531</td>
      <td>18.5618</td>
      <td>1.7202</td>
      <td>25.8398</td>
      <td>4.4354</td>
      <td>3.9642</td>
      <td>3.1364</td>
      <td>12.7803</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>train_1</td>
      <td>0</td>
      <td>11.5006</td>
      <td>-4.1473</td>
      <td>13.8588</td>
      <td>5.6208</td>
      <td>8.0851</td>
      <td>14.0239</td>
      <td>8.4135</td>
      <td>...</td>
      <td>5.7999</td>
      <td>5.5378</td>
      <td>5.5134</td>
      <td>30.2645</td>
      <td>11.0752</td>
      <td>22.5441</td>
      <td>7.6421</td>
      <td>7.7214</td>
      <td>2.5837</td>
      <td>18.3560</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>train_2</td>
      <td>0</td>
      <td>8.6093</td>
      <td>-2.7457</td>
      <td>12.0805</td>
      <td>6.9427</td>
      <td>5.9525</td>
      <td>14.1929</td>
      <td>7.3124</td>
      <td>...</td>
      <td>5.7690</td>
      <td>-7.0927</td>
      <td>-5.8234</td>
      <td>25.6820</td>
      <td>2.4013</td>
      <td>23.0866</td>
      <td>2.9057</td>
      <td>9.7905</td>
      <td>1.6704</td>
      <td>14.7222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>train_3</td>
      <td>0</td>
      <td>11.0604</td>
      <td>-2.1518</td>
      <td>8.9522</td>
      <td>5.8428</td>
      <td>8.2450</td>
      <td>13.8463</td>
      <td>11.9704</td>
      <td>...</td>
      <td>5.3430</td>
      <td>-7.1541</td>
      <td>11.7134</td>
      <td>14.7483</td>
      <td>5.6961</td>
      <td>-0.4639</td>
      <td>4.4666</td>
      <td>4.7433</td>
      <td>0.7178</td>
      <td>17.9697</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>train_4</td>
      <td>0</td>
      <td>9.8369</td>
      <td>-1.4834</td>
      <td>12.8746</td>
      <td>5.9405</td>
      <td>7.6784</td>
      <td>13.8481</td>
      <td>7.8895</td>
      <td>...</td>
      <td>5.5518</td>
      <td>1.4493</td>
      <td>2.3705</td>
      <td>18.4685</td>
      <td>5.1743</td>
      <td>11.8503</td>
      <td>-1.4905</td>
      <td>9.5214</td>
      <td>-0.1508</td>
      <td>17.9974</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3c7d7a08-aefb-49a9-a841-e9561ba6767d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3c7d7a08-aefb-49a9-a841-e9561ba6767d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3c7d7a08-aefb-49a9-a841-e9561ba6767d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### Tail


```python
# last few rows
transaction_df.tail()
```





  <div id="df-9a3ce294-62c9-4107-b61b-fa29d5d5913c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID_code</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>...</th>
      <th>var_40</th>
      <th>var_41</th>
      <th>var_42</th>
      <th>var_43</th>
      <th>var_44</th>
      <th>var_45</th>
      <th>var_46</th>
      <th>var_47</th>
      <th>var_48</th>
      <th>var_49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179995</th>
      <td>179995</td>
      <td>train_179995</td>
      <td>0</td>
      <td>8.9112</td>
      <td>-4.4900</td>
      <td>11.3795</td>
      <td>5.4164</td>
      <td>8.3309</td>
      <td>14.1869</td>
      <td>12.8219</td>
      <td>...</td>
      <td>5.9355</td>
      <td>5.1764</td>
      <td>-5.4817</td>
      <td>20.7390</td>
      <td>-0.8043</td>
      <td>31.3663</td>
      <td>8.7194</td>
      <td>4.7839</td>
      <td>0.9673</td>
      <td>14.9529</td>
    </tr>
    <tr>
      <th>179996</th>
      <td>179996</td>
      <td>train_179996</td>
      <td>0</td>
      <td>8.2722</td>
      <td>7.1923</td>
      <td>16.7300</td>
      <td>4.9698</td>
      <td>5.4907</td>
      <td>14.0238</td>
      <td>11.9933</td>
      <td>...</td>
      <td>5.3550</td>
      <td>-1.5556</td>
      <td>7.3839</td>
      <td>9.9193</td>
      <td>5.0860</td>
      <td>28.0185</td>
      <td>7.4145</td>
      <td>4.2814</td>
      <td>2.5575</td>
      <td>15.3969</td>
    </tr>
    <tr>
      <th>179997</th>
      <td>179997</td>
      <td>train_179997</td>
      <td>0</td>
      <td>12.4902</td>
      <td>-2.9379</td>
      <td>6.5901</td>
      <td>4.7242</td>
      <td>9.1423</td>
      <td>13.9133</td>
      <td>12.2314</td>
      <td>...</td>
      <td>6.0708</td>
      <td>-0.3463</td>
      <td>-5.0098</td>
      <td>20.4807</td>
      <td>5.9814</td>
      <td>20.2668</td>
      <td>3.1006</td>
      <td>3.7701</td>
      <td>0.1299</td>
      <td>16.1256</td>
    </tr>
    <tr>
      <th>179998</th>
      <td>179998</td>
      <td>train_179998</td>
      <td>0</td>
      <td>8.4894</td>
      <td>0.7048</td>
      <td>11.1518</td>
      <td>4.6370</td>
      <td>8.1321</td>
      <td>14.2921</td>
      <td>6.9582</td>
      <td>...</td>
      <td>5.5161</td>
      <td>3.1774</td>
      <td>-4.9374</td>
      <td>23.6365</td>
      <td>1.0432</td>
      <td>29.9365</td>
      <td>9.6381</td>
      <td>6.9649</td>
      <td>2.5546</td>
      <td>15.6760</td>
    </tr>
    <tr>
      <th>179999</th>
      <td>179999</td>
      <td>train_179999</td>
      <td>0</td>
      <td>11.6252</td>
      <td>3.3959</td>
      <td>11.5558</td>
      <td>4.2003</td>
      <td>7.4666</td>
      <td>13.7381</td>
      <td>12.9590</td>
      <td>...</td>
      <td>5.0415</td>
      <td>-2.4210</td>
      <td>-3.6829</td>
      <td>14.2422</td>
      <td>1.0741</td>
      <td>26.3943</td>
      <td>4.1482</td>
      <td>11.9539</td>
      <td>2.9846</td>
      <td>15.6002</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9a3ce294-62c9-4107-b61b-fa29d5d5913c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9a3ce294-62c9-4107-b61b-fa29d5d5913c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9a3ce294-62c9-4107-b61b-fa29d5d5913c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### Shape


```python
# shape of data frame
transaction_df.shape

#output (# of rows, # of columns)
```




    (180000, 53)



### Size


```python
# size of data frame
transaction_df.size
```




    9540000



### Info


```python
# display meta-data
transaction_df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180000 entries, 0 to 179999
    Data columns (total 53 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   Unnamed: 0  180000 non-null  int64  
     1   ID_code     180000 non-null  object 
     2   target      180000 non-null  int64  
     3   var_0       180000 non-null  float64
     4   var_1       180000 non-null  float64
     5   var_2       180000 non-null  float64
     6   var_3       180000 non-null  float64
     7   var_4       180000 non-null  float64
     8   var_5       180000 non-null  float64
     9   var_6       180000 non-null  float64
     10  var_7       180000 non-null  float64
     11  var_8       180000 non-null  float64
     12  var_9       180000 non-null  float64
     13  var_10      180000 non-null  float64
     14  var_11      180000 non-null  float64
     15  var_12      180000 non-null  float64
     16  var_13      180000 non-null  float64
     17  var_14      180000 non-null  float64
     18  var_15      180000 non-null  float64
     19  var_16      180000 non-null  float64
     20  var_17      180000 non-null  float64
     21  var_18      180000 non-null  float64
     22  var_19      180000 non-null  float64
     23  var_20      180000 non-null  float64
     24  var_21      180000 non-null  float64
     25  var_22      180000 non-null  float64
     26  var_23      180000 non-null  float64
     27  var_24      180000 non-null  float64
     28  var_25      180000 non-null  float64
     29  var_26      180000 non-null  float64
     30  var_27      180000 non-null  float64
     31  var_28      180000 non-null  float64
     32  var_29      180000 non-null  float64
     33  var_30      180000 non-null  float64
     34  var_31      180000 non-null  float64
     35  var_32      180000 non-null  float64
     36  var_33      180000 non-null  float64
     37  var_34      180000 non-null  float64
     38  var_35      180000 non-null  float64
     39  var_36      180000 non-null  float64
     40  var_37      180000 non-null  float64
     41  var_38      180000 non-null  float64
     42  var_39      180000 non-null  float64
     43  var_40      180000 non-null  float64
     44  var_41      180000 non-null  float64
     45  var_42      180000 non-null  float64
     46  var_43      180000 non-null  float64
     47  var_44      180000 non-null  float64
     48  var_45      180000 non-null  float64
     49  var_46      180000 non-null  float64
     50  var_47      180000 non-null  float64
     51  var_48      180000 non-null  float64
     52  var_49      180000 non-null  float64
    dtypes: float64(50), int64(2), object(1)
    memory usage: 72.8+ MB


## Summary statistics

### Describe


```python
# display summary statistics
transaction_df.describe()

```





  <div id="df-73464f7c-9d37-4c73-afe8-c0fbbd0f1a58" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>...</th>
      <th>var_40</th>
      <th>var_41</th>
      <th>var_42</th>
      <th>var_43</th>
      <th>var_44</th>
      <th>var_45</th>
      <th>var_46</th>
      <th>var_47</th>
      <th>var_48</th>
      <th>var_49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.00000</td>
      <td>180000.000000</td>
      <td>...</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
      <td>180000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>89999.500000</td>
      <td>0.100222</td>
      <td>10.678650</td>
      <td>-1.630045</td>
      <td>10.712922</td>
      <td>5.408688</td>
      <td>7.566953</td>
      <td>14.024017</td>
      <td>8.52808</td>
      <td>15.243424</td>
      <td>...</td>
      <td>5.635897</td>
      <td>-0.000418</td>
      <td>-0.668334</td>
      <td>20.208845</td>
      <td>2.595090</td>
      <td>12.756574</td>
      <td>3.236201</td>
      <td>7.440715</td>
      <td>1.927078</td>
      <td>15.870166</td>
    </tr>
    <tr>
      <th>std</th>
      <td>51961.668564</td>
      <td>0.300297</td>
      <td>3.041036</td>
      <td>4.050519</td>
      <td>2.639146</td>
      <td>0.866265</td>
      <td>1.235043</td>
      <td>0.190038</td>
      <td>4.64170</td>
      <td>7.851050</td>
      <td>...</td>
      <td>0.369629</td>
      <td>4.422915</td>
      <td>5.968300</td>
      <td>7.132446</td>
      <td>2.801692</td>
      <td>9.320939</td>
      <td>4.561317</td>
      <td>3.021636</td>
      <td>1.476777</td>
      <td>3.010492</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.408400</td>
      <td>-15.043400</td>
      <td>2.117100</td>
      <td>2.347300</td>
      <td>3.970500</td>
      <td>13.434600</td>
      <td>-6.01110</td>
      <td>-10.664200</td>
      <td>...</td>
      <td>4.258800</td>
      <td>-14.506000</td>
      <td>-22.748700</td>
      <td>-2.995300</td>
      <td>-7.856100</td>
      <td>-18.489700</td>
      <td>-14.093300</td>
      <td>-2.691700</td>
      <td>-3.814500</td>
      <td>6.299300</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44999.750000</td>
      <td>0.000000</td>
      <td>8.451675</td>
      <td>-4.745600</td>
      <td>8.720000</td>
      <td>4.768500</td>
      <td>6.618800</td>
      <td>13.893900</td>
      <td>5.06575</td>
      <td>9.170200</td>
      <td>...</td>
      <td>5.374200</td>
      <td>-3.253575</td>
      <td>-4.997500</td>
      <td>15.062900</td>
      <td>0.628600</td>
      <td>5.661300</td>
      <td>-0.062000</td>
      <td>5.160325</td>
      <td>0.890000</td>
      <td>13.830400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>89999.500000</td>
      <td>0.000000</td>
      <td>10.526750</td>
      <td>-1.611250</td>
      <td>10.579600</td>
      <td>5.385600</td>
      <td>7.628500</td>
      <td>14.025500</td>
      <td>8.59920</td>
      <td>15.198900</td>
      <td>...</td>
      <td>5.634500</td>
      <td>0.008500</td>
      <td>-0.560300</td>
      <td>20.203300</td>
      <td>2.513800</td>
      <td>12.520300</td>
      <td>3.204650</td>
      <td>7.350950</td>
      <td>1.899700</td>
      <td>15.934600</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>134999.250000</td>
      <td>0.000000</td>
      <td>12.756200</td>
      <td>1.360225</td>
      <td>12.514100</td>
      <td>6.003200</td>
      <td>8.583800</td>
      <td>14.164400</td>
      <td>12.27330</td>
      <td>21.010500</td>
      <td>...</td>
      <td>5.905600</td>
      <td>3.098325</td>
      <td>3.632525</td>
      <td>25.639125</td>
      <td>4.396725</td>
      <td>19.461725</td>
      <td>6.409900</td>
      <td>9.511125</td>
      <td>2.947225</td>
      <td>18.061425</td>
    </tr>
    <tr>
      <th>max</th>
      <td>179999.000000</td>
      <td>1.000000</td>
      <td>20.315000</td>
      <td>10.376800</td>
      <td>19.353000</td>
      <td>8.447700</td>
      <td>11.150600</td>
      <td>14.654500</td>
      <td>22.33150</td>
      <td>41.748000</td>
      <td>...</td>
      <td>7.089000</td>
      <td>16.731900</td>
      <td>18.855400</td>
      <td>43.546800</td>
      <td>13.448700</td>
      <td>44.363400</td>
      <td>18.440900</td>
      <td>16.716500</td>
      <td>8.402400</td>
      <td>26.079100</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 52 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-73464f7c-9d37-4c73-afe8-c0fbbd0f1a58')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-73464f7c-9d37-4c73-afe8-c0fbbd0f1a58 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-73464f7c-9d37-4c73-afe8-c0fbbd0f1a58');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### Target value counts

NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.


```python
# value_counts on target ; include nulls
transaction_df['target'].value_counts(dropna=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>target</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>161960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18040</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
#normalize value counts on target ; include nulls
transaction_df['target'].value_counts(normalize=True, dropna=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>target</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.899778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.100222</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



### Nulls


```python
# look for nulls
transaction_df.isnull().sum().sort_values(ascending=False)

#output null rows:none (every column has 0 nulls, so no row can contain one)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>ID_code</th>
      <td>0</td>
    </tr>
    <tr>
      <th>target</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_6</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_7</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_8</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_9</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_10</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_11</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_13</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_14</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_15</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_16</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_17</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_18</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_19</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_20</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_21</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_22</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_25</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_27</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_28</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_29</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_30</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_31</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_32</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_33</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_34</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_35</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_36</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_37</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_38</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_39</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_40</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_41</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_42</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_43</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_44</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_45</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_46</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_47</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_48</th>
      <td>0</td>
    </tr>
    <tr>
      <th>var_49</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



### Column data types


```python
# display data types
transaction_df.dtypes.sort_values()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>target</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>var_26</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_27</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_28</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_29</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_30</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_31</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_32</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_33</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_34</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_35</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_36</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_37</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_38</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_39</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_40</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_41</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_42</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_43</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_44</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_45</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_46</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_47</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_25</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_24</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_23</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_22</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_0</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_1</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_2</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_3</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_4</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_5</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_6</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_7</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_8</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_9</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_48</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_10</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_12</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_13</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_14</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_15</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_16</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_17</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_18</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_19</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_20</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_21</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_11</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>var_49</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>ID_code</th>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



###Unique elements


```python
# number of unique elements in every column (works on the whole dataframe)
transaction_df.nunique().sort_values(ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>180000</td>
    </tr>
    <tr>
      <th>ID_code</th>
      <td>180000</td>
    </tr>
    <tr>
      <th>var_35</th>
      <td>137280</td>
    </tr>
    <tr>
      <th>var_45</th>
      <td>134537</td>
    </tr>
    <tr>
      <th>var_13</th>
      <td>131746</td>
    </tr>
    <tr>
      <th>var_21</th>
      <td>130573</td>
    </tr>
    <tr>
      <th>var_8</th>
      <td>130208</td>
    </tr>
    <tr>
      <th>var_7</th>
      <td>129702</td>
    </tr>
    <tr>
      <th>var_17</th>
      <td>129453</td>
    </tr>
    <tr>
      <th>var_33</th>
      <td>128142</td>
    </tr>
    <tr>
      <th>var_16</th>
      <td>127809</td>
    </tr>
    <tr>
      <th>var_31</th>
      <td>127689</td>
    </tr>
    <tr>
      <th>var_19</th>
      <td>127006</td>
    </tr>
    <tr>
      <th>var_43</th>
      <td>125668</td>
    </tr>
    <tr>
      <th>var_42</th>
      <td>119849</td>
    </tr>
    <tr>
      <th>var_14</th>
      <td>119476</td>
    </tr>
    <tr>
      <th>var_10</th>
      <td>118977</td>
    </tr>
    <tr>
      <th>var_37</th>
      <td>115230</td>
    </tr>
    <tr>
      <th>var_29</th>
      <td>114505</td>
    </tr>
    <tr>
      <th>var_36</th>
      <td>112328</td>
    </tr>
    <tr>
      <th>var_38</th>
      <td>112124</td>
    </tr>
    <tr>
      <th>var_6</th>
      <td>108599</td>
    </tr>
    <tr>
      <th>var_46</th>
      <td>108401</td>
    </tr>
    <tr>
      <th>var_41</th>
      <td>107417</td>
    </tr>
    <tr>
      <th>var_11</th>
      <td>106142</td>
    </tr>
    <tr>
      <th>var_26</th>
      <td>106000</td>
    </tr>
    <tr>
      <th>var_22</th>
      <td>104710</td>
    </tr>
    <tr>
      <th>var_1</th>
      <td>103081</td>
    </tr>
    <tr>
      <th>var_27</th>
      <td>100528</td>
    </tr>
    <tr>
      <th>var_0</th>
      <td>90443</td>
    </tr>
    <tr>
      <th>var_47</th>
      <td>89982</td>
    </tr>
    <tr>
      <th>var_49</th>
      <td>89845</td>
    </tr>
    <tr>
      <th>var_9</th>
      <td>86773</td>
    </tr>
    <tr>
      <th>var_44</th>
      <td>86228</td>
    </tr>
    <tr>
      <th>var_23</th>
      <td>85501</td>
    </tr>
    <tr>
      <th>var_28</th>
      <td>83117</td>
    </tr>
    <tr>
      <th>var_2</th>
      <td>83041</td>
    </tr>
    <tr>
      <th>var_32</th>
      <td>80967</td>
    </tr>
    <tr>
      <th>var_20</th>
      <td>76198</td>
    </tr>
    <tr>
      <th>var_18</th>
      <td>69951</td>
    </tr>
    <tr>
      <th>var_24</th>
      <td>67117</td>
    </tr>
    <tr>
      <th>var_48</th>
      <td>57444</td>
    </tr>
    <tr>
      <th>var_4</th>
      <td>48476</td>
    </tr>
    <tr>
      <th>var_3</th>
      <td>37907</td>
    </tr>
    <tr>
      <th>var_15</th>
      <td>32995</td>
    </tr>
    <tr>
      <th>var_12</th>
      <td>24845</td>
    </tr>
    <tr>
      <th>var_30</th>
      <td>19006</td>
    </tr>
    <tr>
      <th>var_40</th>
      <td>18053</td>
    </tr>
    <tr>
      <th>var_39</th>
      <td>17760</td>
    </tr>
    <tr>
      <th>var_34</th>
      <td>10512</td>
    </tr>
    <tr>
      <th>var_5</th>
      <td>9500</td>
    </tr>
    <tr>
      <th>var_25</th>
      <td>8479</td>
    </tr>
    <tr>
      <th>target</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# actual unique values in one column
transaction_df['target'].unique()
```




    array([0, 1])



## Data Cleaning



Backup #1


```python
# backup copy of the original data
transaction_df_backup= transaction_df.copy()

```


```python
# restore from backup
transaction_df=transaction_df_backup.copy()
```

### Keep only the target and features

Selected `target` for the target and `var_0 through var_49` for the features


```python
# select the target (All rows= 180,000)
transaction_df_target=transaction_df[['target']].head(180000).copy()
transaction_df_target.shape
```




    (180000, 1)




```python
# select the features
feature_cols =[] #start with an empty list to hold the features
for col in transaction_df.columns: #look at each column name
  if col.startswith ('var_'): #if the name start with 'var_, it's a feature
    feature_cols.append(col)
# check the list (show 50 names)
print(feature_cols)
print(len(feature_cols))
```

    ['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15', 'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23', 'var_24', 'var_25', 'var_26', 'var_27', 'var_28', 'var_29', 'var_30', 'var_31', 'var_32', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38', 'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_45', 'var_46', 'var_47', 'var_48', 'var_49']
    50



```python
# select the target and the features
#feature_cols= feature_cols #using all the column dont need this code line

#keep all rows 180,000=(.head(# of rows))
transaction_df_target_feature = transaction_df[['target'] + feature_cols].head(180000).copy()

transaction_df_target_feature.shape #verify

```




    (180000, 51)



## Recommendations for Data Cleaning

**1. Identifier Columns**
- **Two columns:** `Unnamed: 0` is row numbers and `ID_code` is customer ID. cannot help predict anything
- **Recommendation:** Drop both columns. (I kept only `target` and `var_0` to `var_49`.)

**2. Null values:** No feature column have nulls. no action needed

**3. Target nulls:** none, no rows dropped

**4. Feature columns with high null counts:** None

**5. Medium null counts:** none

**6. Low null counts:** none

**7. Null rows:** none


## Exploratory Data Analysis ##

### Normal distribution of features:histogram


```python
# Display a histogram - overview of target and the all features

transaction_df_target_feature.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

```


    
![png](Project.1-Pass-04_files/Project.1-Pass-04_47_0.png)
    



```python
# Check the distributions of the all features (var_0 to var_49)

transaction_df_target_feature[feature_cols].hist(
  figsize=(15,10), # size of the whole picture in inches(width,height)
  layout=(7,8),
  bins=30,         # more bars, so the shape is clear
  edgecolor= "red"
)
plt.tight_layout()
plt.show()
#output- each graph look like a bell-shaped (first GNB assumption)
```


    
![png](Project.1-Pass-04_files/Project.1-Pass-04_48_0.png)
    


### Independence of features: Correlation matrix


```python
# Calculate a correlation matrix (target & features)
transaction_df_target_feature.corr()
```





  <div id="df-fd49d7f2-b036-4520-aad4-710c95b91668" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>var_0</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>var_8</th>
      <th>...</th>
      <th>var_40</th>
      <th>var_41</th>
      <th>var_42</th>
      <th>var_43</th>
      <th>var_44</th>
      <th>var_45</th>
      <th>var_46</th>
      <th>var_47</th>
      <th>var_48</th>
      <th>var_49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>1.000000</td>
      <td>0.052904</td>
      <td>0.049916</td>
      <td>0.055450</td>
      <td>0.066924</td>
      <td>-0.042073</td>
      <td>-0.070015</td>
      <td>-0.054375</td>
      <td>0.044769</td>
      <td>-0.057378</td>
      <td>...</td>
      <td>-0.050049</td>
      <td>0.047948</td>
      <td>0.041593</td>
      <td>-0.061291</td>
      <td>0.051110</td>
      <td>0.047543</td>
      <td>0.056902</td>
      <td>0.045506</td>
      <td>-0.042765</td>
      <td>-0.051718</td>
    </tr>
    <tr>
      <th>var_0</th>
      <td>0.052904</td>
      <td>1.000000</td>
      <td>-0.002155</td>
      <td>0.006632</td>
      <td>0.007155</td>
      <td>-0.002247</td>
      <td>-0.003147</td>
      <td>-0.001717</td>
      <td>0.004266</td>
      <td>0.000289</td>
      <td>...</td>
      <td>-0.002602</td>
      <td>0.002043</td>
      <td>0.001072</td>
      <td>-0.001493</td>
      <td>0.003662</td>
      <td>-0.000227</td>
      <td>0.002330</td>
      <td>-0.000037</td>
      <td>-0.005759</td>
      <td>-0.005945</td>
    </tr>
    <tr>
      <th>var_1</th>
      <td>0.049916</td>
      <td>-0.002155</td>
      <td>1.000000</td>
      <td>0.005644</td>
      <td>0.002640</td>
      <td>-0.000408</td>
      <td>-0.002629</td>
      <td>-0.001234</td>
      <td>-0.000374</td>
      <td>-0.000378</td>
      <td>...</td>
      <td>-0.003109</td>
      <td>0.001891</td>
      <td>0.003331</td>
      <td>-0.005291</td>
      <td>0.000805</td>
      <td>0.000387</td>
      <td>0.005835</td>
      <td>0.003845</td>
      <td>-0.003810</td>
      <td>-0.005352</td>
    </tr>
    <tr>
      <th>var_2</th>
      <td>0.055450</td>
      <td>0.006632</td>
      <td>0.005644</td>
      <td>1.000000</td>
      <td>0.000679</td>
      <td>-0.001293</td>
      <td>-0.002718</td>
      <td>-0.009171</td>
      <td>0.003436</td>
      <td>-0.006433</td>
      <td>...</td>
      <td>-0.004639</td>
      <td>0.002059</td>
      <td>0.000459</td>
      <td>-0.005709</td>
      <td>0.001388</td>
      <td>0.004021</td>
      <td>-0.000657</td>
      <td>0.000968</td>
      <td>-0.002388</td>
      <td>-0.000372</td>
    </tr>
    <tr>
      <th>var_3</th>
      <td>0.066924</td>
      <td>0.007155</td>
      <td>0.002640</td>
      <td>0.000679</td>
      <td>1.000000</td>
      <td>-0.006768</td>
      <td>-0.002568</td>
      <td>-0.005282</td>
      <td>0.002685</td>
      <td>-0.004835</td>
      <td>...</td>
      <td>-0.004825</td>
      <td>0.002188</td>
      <td>-0.000344</td>
      <td>-0.005121</td>
      <td>-0.000999</td>
      <td>0.003463</td>
      <td>0.000276</td>
      <td>0.005311</td>
      <td>-0.001109</td>
      <td>-0.002987</td>
    </tr>
    <tr>
      <th>var_4</th>
      <td>-0.042073</td>
      <td>-0.002247</td>
      <td>-0.000408</td>
      <td>-0.001293</td>
      <td>-0.006768</td>
      <td>1.000000</td>
      <td>0.000845</td>
      <td>0.000210</td>
      <td>-0.001267</td>
      <td>0.000962</td>
      <td>...</td>
      <td>-0.000824</td>
      <td>-0.000685</td>
      <td>-0.004266</td>
      <td>0.000483</td>
      <td>-0.003563</td>
      <td>-0.004458</td>
      <td>0.000043</td>
      <td>-0.000716</td>
      <td>0.000190</td>
      <td>0.001484</td>
    </tr>
    <tr>
      <th>var_5</th>
      <td>-0.070015</td>
      <td>-0.003147</td>
      <td>-0.002629</td>
      <td>-0.002718</td>
      <td>-0.002568</td>
      <td>0.000845</td>
      <td>1.000000</td>
      <td>0.004778</td>
      <td>0.000110</td>
      <td>0.005587</td>
      <td>...</td>
      <td>0.007054</td>
      <td>-0.000356</td>
      <td>-0.006139</td>
      <td>0.004441</td>
      <td>-0.005371</td>
      <td>-0.002999</td>
      <td>0.001721</td>
      <td>-0.004525</td>
      <td>0.003204</td>
      <td>0.008009</td>
    </tr>
    <tr>
      <th>var_6</th>
      <td>-0.054375</td>
      <td>-0.001717</td>
      <td>-0.001234</td>
      <td>-0.009171</td>
      <td>-0.005282</td>
      <td>0.000210</td>
      <td>0.004778</td>
      <td>1.000000</td>
      <td>0.001922</td>
      <td>0.001879</td>
      <td>...</td>
      <td>0.001306</td>
      <td>-0.001122</td>
      <td>-0.001814</td>
      <td>0.000897</td>
      <td>-0.005164</td>
      <td>-0.002714</td>
      <td>-0.001001</td>
      <td>-0.002799</td>
      <td>-0.002182</td>
      <td>-0.001663</td>
    </tr>
    <tr>
      <th>var_7</th>
      <td>0.044769</td>
      <td>0.004266</td>
      <td>-0.000374</td>
      <td>0.003436</td>
      <td>0.002685</td>
      <td>-0.001267</td>
      <td>0.000110</td>
      <td>0.001922</td>
      <td>1.000000</td>
      <td>-0.003654</td>
      <td>...</td>
      <td>-0.000989</td>
      <td>-0.000584</td>
      <td>0.002339</td>
      <td>0.001428</td>
      <td>0.005357</td>
      <td>-0.000275</td>
      <td>0.002816</td>
      <td>0.004656</td>
      <td>-0.006991</td>
      <td>-0.004983</td>
    </tr>
    <tr>
      <th>var_8</th>
      <td>-0.057378</td>
      <td>0.000289</td>
      <td>-0.000378</td>
      <td>-0.006433</td>
      <td>-0.004835</td>
      <td>0.000962</td>
      <td>0.005587</td>
      <td>0.001879</td>
      <td>-0.003654</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.006917</td>
      <td>-0.000172</td>
      <td>-0.004523</td>
      <td>0.003021</td>
      <td>-0.003809</td>
      <td>-0.001034</td>
      <td>-0.003210</td>
      <td>-0.001933</td>
      <td>0.000192</td>
      <td>0.000810</td>
    </tr>
    <tr>
      <th>var_9</th>
      <td>0.059227</td>
      <td>0.002817</td>
      <td>0.001282</td>
      <td>0.000511</td>
      <td>0.002293</td>
      <td>-0.005332</td>
      <td>-0.004708</td>
      <td>0.000293</td>
      <td>0.004690</td>
      <td>-0.004270</td>
      <td>...</td>
      <td>-0.003959</td>
      <td>0.001460</td>
      <td>0.001234</td>
      <td>-0.006182</td>
      <td>-0.000563</td>
      <td>0.001718</td>
      <td>0.004172</td>
      <td>-0.001395</td>
      <td>-0.001118</td>
      <td>-0.005218</td>
    </tr>
    <tr>
      <th>var_10</th>
      <td>0.063565</td>
      <td>0.006102</td>
      <td>0.004161</td>
      <td>0.006676</td>
      <td>0.003590</td>
      <td>-0.003446</td>
      <td>-0.006633</td>
      <td>-0.007201</td>
      <td>0.001987</td>
      <td>-0.002853</td>
      <td>...</td>
      <td>-0.002125</td>
      <td>0.000180</td>
      <td>-0.000539</td>
      <td>-0.006401</td>
      <td>0.001067</td>
      <td>0.004700</td>
      <td>0.001844</td>
      <td>-0.000582</td>
      <td>0.000946</td>
      <td>-0.005822</td>
    </tr>
    <tr>
      <th>var_11</th>
      <td>-0.045516</td>
      <td>-0.003080</td>
      <td>0.000955</td>
      <td>0.001006</td>
      <td>0.000116</td>
      <td>0.003206</td>
      <td>0.008232</td>
      <td>0.002331</td>
      <td>-0.001832</td>
      <td>0.003557</td>
      <td>...</td>
      <td>-0.000445</td>
      <td>-0.000227</td>
      <td>0.003816</td>
      <td>0.004107</td>
      <td>-0.006338</td>
      <td>-0.003121</td>
      <td>-0.002500</td>
      <td>-0.001531</td>
      <td>0.002908</td>
      <td>0.001471</td>
    </tr>
    <tr>
      <th>var_12</th>
      <td>-0.052324</td>
      <td>-0.003419</td>
      <td>-0.003251</td>
      <td>-0.006361</td>
      <td>-0.003659</td>
      <td>0.003278</td>
      <td>0.003767</td>
      <td>0.005578</td>
      <td>-0.005472</td>
      <td>0.002605</td>
      <td>...</td>
      <td>0.002819</td>
      <td>-0.004693</td>
      <td>0.000657</td>
      <td>0.003434</td>
      <td>-0.003245</td>
      <td>0.000097</td>
      <td>-0.003375</td>
      <td>-0.004960</td>
      <td>0.000980</td>
      <td>0.004265</td>
    </tr>
    <tr>
      <th>var_13</th>
      <td>0.049072</td>
      <td>0.006984</td>
      <td>0.001400</td>
      <td>0.000502</td>
      <td>0.000999</td>
      <td>-0.000803</td>
      <td>-0.004156</td>
      <td>-0.004100</td>
      <td>0.000639</td>
      <td>-0.002722</td>
      <td>...</td>
      <td>-0.004848</td>
      <td>0.001784</td>
      <td>-0.000468</td>
      <td>-0.003950</td>
      <td>0.005755</td>
      <td>-0.000328</td>
      <td>0.003868</td>
      <td>0.001837</td>
      <td>-0.003036</td>
      <td>-0.003682</td>
    </tr>
    <tr>
      <th>var_14</th>
      <td>-0.048623</td>
      <td>-0.003111</td>
      <td>-0.005702</td>
      <td>0.001173</td>
      <td>-0.006827</td>
      <td>0.001561</td>
      <td>-0.001322</td>
      <td>0.007995</td>
      <td>0.000424</td>
      <td>0.001981</td>
      <td>...</td>
      <td>-0.000613</td>
      <td>-0.001888</td>
      <td>-0.002119</td>
      <td>0.003086</td>
      <td>-0.004185</td>
      <td>-0.001590</td>
      <td>-0.006136</td>
      <td>-0.004908</td>
      <td>0.001897</td>
      <td>0.003660</td>
    </tr>
    <tr>
      <th>var_15</th>
      <td>0.062887</td>
      <td>-0.000807</td>
      <td>0.004442</td>
      <td>0.000238</td>
      <td>0.003167</td>
      <td>-0.001129</td>
      <td>-0.004501</td>
      <td>-0.000941</td>
      <td>0.006694</td>
      <td>-0.007377</td>
      <td>...</td>
      <td>-0.001949</td>
      <td>0.001232</td>
      <td>0.004893</td>
      <td>-0.004943</td>
      <td>0.001798</td>
      <td>0.005322</td>
      <td>0.005114</td>
      <td>0.002503</td>
      <td>-0.003845</td>
      <td>-0.003435</td>
    </tr>
    <tr>
      <th>var_16</th>
      <td>0.045281</td>
      <td>0.002762</td>
      <td>0.002061</td>
      <td>-0.002321</td>
      <td>0.001771</td>
      <td>-0.003054</td>
      <td>-0.000761</td>
      <td>-0.004249</td>
      <td>0.004549</td>
      <td>-0.003999</td>
      <td>...</td>
      <td>-0.007425</td>
      <td>-0.000849</td>
      <td>0.001377</td>
      <td>-0.002490</td>
      <td>-0.001727</td>
      <td>-0.000160</td>
      <td>0.001755</td>
      <td>0.002901</td>
      <td>-0.004046</td>
      <td>-0.006180</td>
    </tr>
    <tr>
      <th>var_17</th>
      <td>-0.059906</td>
      <td>-0.007308</td>
      <td>-0.003249</td>
      <td>-0.008054</td>
      <td>-0.003353</td>
      <td>-0.000039</td>
      <td>0.005801</td>
      <td>0.001999</td>
      <td>-0.003245</td>
      <td>0.000628</td>
      <td>...</td>
      <td>0.003433</td>
      <td>-0.005842</td>
      <td>0.002917</td>
      <td>0.006450</td>
      <td>-0.001501</td>
      <td>-0.002355</td>
      <td>-0.004609</td>
      <td>-0.002737</td>
      <td>0.004096</td>
      <td>-0.001818</td>
    </tr>
    <tr>
      <th>var_18</th>
      <td>0.048392</td>
      <td>0.003033</td>
      <td>0.002386</td>
      <td>0.003295</td>
      <td>0.005542</td>
      <td>-0.002726</td>
      <td>-0.005806</td>
      <td>-0.003728</td>
      <td>-0.003520</td>
      <td>-0.005804</td>
      <td>...</td>
      <td>-0.004201</td>
      <td>0.002280</td>
      <td>0.004985</td>
      <td>0.000019</td>
      <td>0.001497</td>
      <td>-0.000511</td>
      <td>0.000786</td>
      <td>0.001374</td>
      <td>0.000999</td>
      <td>0.000974</td>
    </tr>
    <tr>
      <th>var_19</th>
      <td>-0.057320</td>
      <td>-0.003159</td>
      <td>-0.009469</td>
      <td>-0.008065</td>
      <td>-0.008879</td>
      <td>0.002086</td>
      <td>0.007575</td>
      <td>0.003756</td>
      <td>-0.005636</td>
      <td>0.006147</td>
      <td>...</td>
      <td>0.000572</td>
      <td>-0.004176</td>
      <td>-0.002271</td>
      <td>-0.001866</td>
      <td>-0.004877</td>
      <td>-0.001920</td>
      <td>-0.003538</td>
      <td>-0.001149</td>
      <td>0.003110</td>
      <td>0.003272</td>
    </tr>
    <tr>
      <th>var_20</th>
      <td>-0.080689</td>
      <td>-0.007543</td>
      <td>-0.002229</td>
      <td>-0.003277</td>
      <td>-0.003882</td>
      <td>0.000842</td>
      <td>0.005668</td>
      <td>0.005505</td>
      <td>-0.004657</td>
      <td>0.005497</td>
      <td>...</td>
      <td>0.001826</td>
      <td>-0.002335</td>
      <td>-0.003469</td>
      <td>0.007864</td>
      <td>-0.004475</td>
      <td>-0.003804</td>
      <td>-0.003004</td>
      <td>-0.000780</td>
      <td>0.005639</td>
      <td>0.004288</td>
    </tr>
    <tr>
      <th>var_21</th>
      <td>-0.040311</td>
      <td>-0.000473</td>
      <td>-0.003555</td>
      <td>-0.004516</td>
      <td>-0.005799</td>
      <td>0.001610</td>
      <td>0.002973</td>
      <td>0.001476</td>
      <td>-0.005194</td>
      <td>-0.000699</td>
      <td>...</td>
      <td>0.000535</td>
      <td>-0.000065</td>
      <td>-0.000539</td>
      <td>0.006382</td>
      <td>-0.001717</td>
      <td>0.001186</td>
      <td>-0.000281</td>
      <td>-0.001353</td>
      <td>0.001449</td>
      <td>0.001590</td>
    </tr>
    <tr>
      <th>var_22</th>
      <td>-0.046433</td>
      <td>0.000999</td>
      <td>0.000353</td>
      <td>-0.001183</td>
      <td>-0.004269</td>
      <td>0.001980</td>
      <td>0.006521</td>
      <td>0.006360</td>
      <td>0.001788</td>
      <td>0.004102</td>
      <td>...</td>
      <td>0.004728</td>
      <td>-0.004615</td>
      <td>-0.003427</td>
      <td>0.004096</td>
      <td>-0.005454</td>
      <td>-0.001161</td>
      <td>-0.003808</td>
      <td>-0.000225</td>
      <td>0.002087</td>
      <td>0.006064</td>
    </tr>
    <tr>
      <th>var_23</th>
      <td>0.047569</td>
      <td>0.002101</td>
      <td>0.003639</td>
      <td>-0.002632</td>
      <td>0.001711</td>
      <td>-0.007927</td>
      <td>-0.003263</td>
      <td>-0.003612</td>
      <td>0.000279</td>
      <td>0.000833</td>
      <td>...</td>
      <td>0.003721</td>
      <td>0.003747</td>
      <td>0.001406</td>
      <td>-0.001695</td>
      <td>0.004777</td>
      <td>0.006113</td>
      <td>0.005556</td>
      <td>0.003831</td>
      <td>-0.002168</td>
      <td>-0.003101</td>
    </tr>
    <tr>
      <th>var_24</th>
      <td>0.059306</td>
      <td>0.004061</td>
      <td>0.004988</td>
      <td>0.005903</td>
      <td>0.005080</td>
      <td>-0.003609</td>
      <td>-0.009386</td>
      <td>-0.002642</td>
      <td>0.000857</td>
      <td>-0.003458</td>
      <td>...</td>
      <td>-0.002736</td>
      <td>-0.000634</td>
      <td>0.000254</td>
      <td>-0.002020</td>
      <td>0.001817</td>
      <td>0.003375</td>
      <td>0.005298</td>
      <td>0.000236</td>
      <td>0.001352</td>
      <td>-0.006188</td>
    </tr>
    <tr>
      <th>var_25</th>
      <td>-0.047005</td>
      <td>-0.003717</td>
      <td>-0.001364</td>
      <td>-0.000660</td>
      <td>-0.002887</td>
      <td>0.004553</td>
      <td>0.003363</td>
      <td>-0.000822</td>
      <td>-0.005090</td>
      <td>0.002765</td>
      <td>...</td>
      <td>0.001780</td>
      <td>-0.005980</td>
      <td>-0.002807</td>
      <td>0.002351</td>
      <td>-0.002234</td>
      <td>0.000085</td>
      <td>-0.007496</td>
      <td>0.001475</td>
      <td>0.001439</td>
      <td>0.003964</td>
    </tr>
    <tr>
      <th>var_26</th>
      <td>-0.050278</td>
      <td>-0.004076</td>
      <td>-0.000936</td>
      <td>-0.000789</td>
      <td>-0.001757</td>
      <td>0.002389</td>
      <td>0.005223</td>
      <td>0.000993</td>
      <td>-0.004584</td>
      <td>0.000030</td>
      <td>...</td>
      <td>0.004112</td>
      <td>-0.006242</td>
      <td>-0.003807</td>
      <td>0.005348</td>
      <td>-0.005194</td>
      <td>-0.000124</td>
      <td>0.002478</td>
      <td>-0.003738</td>
      <td>-0.000693</td>
      <td>0.004289</td>
    </tr>
    <tr>
      <th>var_27</th>
      <td>0.064868</td>
      <td>0.006209</td>
      <td>0.002923</td>
      <td>0.004149</td>
      <td>0.003192</td>
      <td>-0.002513</td>
      <td>-0.004910</td>
      <td>-0.001736</td>
      <td>0.000274</td>
      <td>-0.001332</td>
      <td>...</td>
      <td>-0.004676</td>
      <td>0.004848</td>
      <td>0.006782</td>
      <td>-0.003821</td>
      <td>-0.000188</td>
      <td>0.002833</td>
      <td>0.002766</td>
      <td>0.006135</td>
      <td>-0.000717</td>
      <td>0.001540</td>
    </tr>
    <tr>
      <th>var_28</th>
      <td>-0.049551</td>
      <td>-0.000320</td>
      <td>-0.007531</td>
      <td>-0.002385</td>
      <td>-0.004857</td>
      <td>0.001646</td>
      <td>0.002953</td>
      <td>-0.000732</td>
      <td>-0.004421</td>
      <td>0.002168</td>
      <td>...</td>
      <td>-0.001076</td>
      <td>-0.003796</td>
      <td>-0.001616</td>
      <td>0.001203</td>
      <td>-0.004178</td>
      <td>-0.002877</td>
      <td>-0.001766</td>
      <td>0.000636</td>
      <td>0.003358</td>
      <td>0.002573</td>
    </tr>
    <tr>
      <th>var_29</th>
      <td>-0.041987</td>
      <td>-0.000785</td>
      <td>-0.003454</td>
      <td>-0.003774</td>
      <td>-0.000245</td>
      <td>0.002574</td>
      <td>0.002366</td>
      <td>0.004282</td>
      <td>-0.001980</td>
      <td>0.000514</td>
      <td>...</td>
      <td>0.001570</td>
      <td>-0.005960</td>
      <td>-0.002750</td>
      <td>0.004144</td>
      <td>-0.003721</td>
      <td>-0.001824</td>
      <td>-0.004441</td>
      <td>-0.001320</td>
      <td>0.002811</td>
      <td>0.000783</td>
    </tr>
    <tr>
      <th>var_30</th>
      <td>0.052457</td>
      <td>0.002141</td>
      <td>0.002295</td>
      <td>0.000987</td>
      <td>0.002256</td>
      <td>-0.002446</td>
      <td>-0.003349</td>
      <td>-0.002727</td>
      <td>0.005089</td>
      <td>-0.003145</td>
      <td>...</td>
      <td>-0.004834</td>
      <td>0.004496</td>
      <td>0.003347</td>
      <td>-0.002294</td>
      <td>-0.000577</td>
      <td>0.001117</td>
      <td>0.001007</td>
      <td>0.002949</td>
      <td>-0.003929</td>
      <td>-0.007127</td>
    </tr>
    <tr>
      <th>var_31</th>
      <td>-0.073296</td>
      <td>-0.006605</td>
      <td>0.001009</td>
      <td>-0.002583</td>
      <td>-0.003911</td>
      <td>0.000657</td>
      <td>0.004910</td>
      <td>0.005050</td>
      <td>-0.000393</td>
      <td>0.000611</td>
      <td>...</td>
      <td>0.005624</td>
      <td>-0.007350</td>
      <td>-0.001621</td>
      <td>0.005679</td>
      <td>-0.004172</td>
      <td>-0.005207</td>
      <td>-0.002300</td>
      <td>-0.004745</td>
      <td>-0.001672</td>
      <td>0.004164</td>
    </tr>
    <tr>
      <th>var_32</th>
      <td>-0.063871</td>
      <td>0.000216</td>
      <td>-0.004563</td>
      <td>-0.003880</td>
      <td>-0.001359</td>
      <td>-0.000499</td>
      <td>0.008401</td>
      <td>0.004149</td>
      <td>-0.000616</td>
      <td>0.002166</td>
      <td>...</td>
      <td>0.009601</td>
      <td>-0.002546</td>
      <td>-0.004211</td>
      <td>0.003646</td>
      <td>-0.004382</td>
      <td>-0.005613</td>
      <td>-0.005490</td>
      <td>-0.004664</td>
      <td>0.007556</td>
      <td>-0.000485</td>
    </tr>
    <tr>
      <th>var_33</th>
      <td>0.040520</td>
      <td>0.001071</td>
      <td>0.002160</td>
      <td>0.001614</td>
      <td>-0.004274</td>
      <td>-0.002492</td>
      <td>0.000237</td>
      <td>-0.001968</td>
      <td>0.005070</td>
      <td>-0.000502</td>
      <td>...</td>
      <td>-0.003602</td>
      <td>-0.000380</td>
      <td>0.001090</td>
      <td>-0.003304</td>
      <td>0.001407</td>
      <td>0.004643</td>
      <td>0.003803</td>
      <td>0.001668</td>
      <td>0.002127</td>
      <td>0.000071</td>
    </tr>
    <tr>
      <th>var_34</th>
      <td>-0.053733</td>
      <td>-0.001549</td>
      <td>-0.001535</td>
      <td>-0.000990</td>
      <td>-0.006517</td>
      <td>0.004187</td>
      <td>0.006401</td>
      <td>0.001237</td>
      <td>-0.000707</td>
      <td>-0.001145</td>
      <td>...</td>
      <td>0.005144</td>
      <td>-0.004184</td>
      <td>-0.005267</td>
      <td>0.004082</td>
      <td>-0.004755</td>
      <td>-0.003125</td>
      <td>0.002361</td>
      <td>-0.005451</td>
      <td>0.001290</td>
      <td>0.002411</td>
    </tr>
    <tr>
      <th>var_35</th>
      <td>-0.048198</td>
      <td>-0.003448</td>
      <td>-0.003538</td>
      <td>-0.005730</td>
      <td>-0.002966</td>
      <td>0.002901</td>
      <td>0.003790</td>
      <td>0.004320</td>
      <td>-0.005347</td>
      <td>0.003576</td>
      <td>...</td>
      <td>0.007536</td>
      <td>-0.001162</td>
      <td>-0.004803</td>
      <td>0.002481</td>
      <td>-0.002155</td>
      <td>-0.003571</td>
      <td>-0.000975</td>
      <td>-0.001241</td>
      <td>0.001802</td>
      <td>0.004302</td>
    </tr>
    <tr>
      <th>var_36</th>
      <td>-0.045513</td>
      <td>-0.002529</td>
      <td>0.000531</td>
      <td>-0.005019</td>
      <td>-0.003190</td>
      <td>0.002981</td>
      <td>0.002574</td>
      <td>0.007089</td>
      <td>0.001718</td>
      <td>0.005592</td>
      <td>...</td>
      <td>0.006974</td>
      <td>-0.004606</td>
      <td>-0.001116</td>
      <td>0.000092</td>
      <td>-0.003130</td>
      <td>-0.005077</td>
      <td>-0.005069</td>
      <td>0.000087</td>
      <td>0.001005</td>
      <td>0.007019</td>
    </tr>
    <tr>
      <th>var_37</th>
      <td>0.040147</td>
      <td>-0.000680</td>
      <td>0.003900</td>
      <td>-0.000033</td>
      <td>0.001802</td>
      <td>0.001161</td>
      <td>-0.002633</td>
      <td>-0.005349</td>
      <td>0.003749</td>
      <td>0.001829</td>
      <td>...</td>
      <td>-0.004036</td>
      <td>0.004512</td>
      <td>0.005514</td>
      <td>-0.001162</td>
      <td>0.004344</td>
      <td>-0.000635</td>
      <td>0.002209</td>
      <td>-0.000477</td>
      <td>-0.005028</td>
      <td>-0.004615</td>
    </tr>
    <tr>
      <th>var_38</th>
      <td>-0.055704</td>
      <td>-0.002785</td>
      <td>0.002074</td>
      <td>-0.003032</td>
      <td>0.001561</td>
      <td>0.004357</td>
      <td>0.004442</td>
      <td>0.005744</td>
      <td>-0.001795</td>
      <td>0.006263</td>
      <td>...</td>
      <td>0.002589</td>
      <td>-0.004132</td>
      <td>-0.002000</td>
      <td>0.000845</td>
      <td>-0.002044</td>
      <td>-0.005268</td>
      <td>-0.006343</td>
      <td>0.000383</td>
      <td>0.002691</td>
      <td>0.003957</td>
    </tr>
    <tr>
      <th>var_39</th>
      <td>-0.057373</td>
      <td>-0.008580</td>
      <td>-0.007379</td>
      <td>-0.008062</td>
      <td>-0.003894</td>
      <td>0.004569</td>
      <td>0.007981</td>
      <td>0.002467</td>
      <td>-0.002751</td>
      <td>0.001496</td>
      <td>...</td>
      <td>0.005273</td>
      <td>-0.000018</td>
      <td>-0.004732</td>
      <td>0.004092</td>
      <td>-0.003692</td>
      <td>-0.001573</td>
      <td>-0.000609</td>
      <td>-0.007758</td>
      <td>0.003066</td>
      <td>-0.002585</td>
    </tr>
    <tr>
      <th>var_40</th>
      <td>-0.050049</td>
      <td>-0.002602</td>
      <td>-0.003109</td>
      <td>-0.004639</td>
      <td>-0.004825</td>
      <td>-0.000824</td>
      <td>0.007054</td>
      <td>0.001306</td>
      <td>-0.000989</td>
      <td>0.006917</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.002300</td>
      <td>-0.003330</td>
      <td>0.004815</td>
      <td>-0.001172</td>
      <td>-0.003808</td>
      <td>-0.002916</td>
      <td>-0.006360</td>
      <td>0.000607</td>
      <td>0.002123</td>
    </tr>
    <tr>
      <th>var_41</th>
      <td>0.047948</td>
      <td>0.002043</td>
      <td>0.001891</td>
      <td>0.002059</td>
      <td>0.002188</td>
      <td>-0.000685</td>
      <td>-0.000356</td>
      <td>-0.001122</td>
      <td>-0.000584</td>
      <td>-0.000172</td>
      <td>...</td>
      <td>-0.002300</td>
      <td>1.000000</td>
      <td>0.002092</td>
      <td>-0.004860</td>
      <td>-0.000794</td>
      <td>-0.000843</td>
      <td>0.004609</td>
      <td>0.001009</td>
      <td>-0.003694</td>
      <td>0.000379</td>
    </tr>
    <tr>
      <th>var_42</th>
      <td>0.041593</td>
      <td>0.001072</td>
      <td>0.003331</td>
      <td>0.000459</td>
      <td>-0.000344</td>
      <td>-0.004266</td>
      <td>-0.006139</td>
      <td>-0.001814</td>
      <td>0.002339</td>
      <td>-0.004523</td>
      <td>...</td>
      <td>-0.003330</td>
      <td>0.002092</td>
      <td>1.000000</td>
      <td>-0.002379</td>
      <td>0.001463</td>
      <td>-0.004179</td>
      <td>-0.001474</td>
      <td>-0.000269</td>
      <td>-0.002330</td>
      <td>-0.000821</td>
    </tr>
    <tr>
      <th>var_43</th>
      <td>-0.061291</td>
      <td>-0.001493</td>
      <td>-0.005291</td>
      <td>-0.005709</td>
      <td>-0.005121</td>
      <td>0.000483</td>
      <td>0.004441</td>
      <td>0.000897</td>
      <td>0.001428</td>
      <td>0.003021</td>
      <td>...</td>
      <td>0.004815</td>
      <td>-0.004860</td>
      <td>-0.002379</td>
      <td>1.000000</td>
      <td>-0.001271</td>
      <td>-0.000705</td>
      <td>-0.004706</td>
      <td>-0.004553</td>
      <td>0.002030</td>
      <td>0.002975</td>
    </tr>
    <tr>
      <th>var_44</th>
      <td>0.051110</td>
      <td>0.003662</td>
      <td>0.000805</td>
      <td>0.001388</td>
      <td>-0.000999</td>
      <td>-0.003563</td>
      <td>-0.005371</td>
      <td>-0.005164</td>
      <td>0.005357</td>
      <td>-0.003809</td>
      <td>...</td>
      <td>-0.001172</td>
      <td>-0.000794</td>
      <td>0.001463</td>
      <td>-0.001271</td>
      <td>1.000000</td>
      <td>-0.000230</td>
      <td>0.002904</td>
      <td>0.006413</td>
      <td>0.000092</td>
      <td>0.000081</td>
    </tr>
    <tr>
      <th>var_45</th>
      <td>0.047543</td>
      <td>-0.000227</td>
      <td>0.000387</td>
      <td>0.004021</td>
      <td>0.003463</td>
      <td>-0.004458</td>
      <td>-0.002999</td>
      <td>-0.002714</td>
      <td>-0.000275</td>
      <td>-0.001034</td>
      <td>...</td>
      <td>-0.003808</td>
      <td>-0.000843</td>
      <td>-0.004179</td>
      <td>-0.000705</td>
      <td>-0.000230</td>
      <td>1.000000</td>
      <td>0.004576</td>
      <td>0.004577</td>
      <td>-0.000009</td>
      <td>-0.003859</td>
    </tr>
    <tr>
      <th>var_46</th>
      <td>0.056902</td>
      <td>0.002330</td>
      <td>0.005835</td>
      <td>-0.000657</td>
      <td>0.000276</td>
      <td>0.000043</td>
      <td>0.001721</td>
      <td>-0.001001</td>
      <td>0.002816</td>
      <td>-0.003210</td>
      <td>...</td>
      <td>-0.002916</td>
      <td>0.004609</td>
      <td>-0.001474</td>
      <td>-0.004706</td>
      <td>0.002904</td>
      <td>0.004576</td>
      <td>1.000000</td>
      <td>0.002885</td>
      <td>0.001399</td>
      <td>-0.001607</td>
    </tr>
    <tr>
      <th>var_47</th>
      <td>0.045506</td>
      <td>-0.000037</td>
      <td>0.003845</td>
      <td>0.000968</td>
      <td>0.005311</td>
      <td>-0.000716</td>
      <td>-0.004525</td>
      <td>-0.002799</td>
      <td>0.004656</td>
      <td>-0.001933</td>
      <td>...</td>
      <td>-0.006360</td>
      <td>0.001009</td>
      <td>-0.000269</td>
      <td>-0.004553</td>
      <td>0.006413</td>
      <td>0.004577</td>
      <td>0.002885</td>
      <td>1.000000</td>
      <td>-0.001088</td>
      <td>-0.000892</td>
    </tr>
    <tr>
      <th>var_48</th>
      <td>-0.042765</td>
      <td>-0.005759</td>
      <td>-0.003810</td>
      <td>-0.002388</td>
      <td>-0.001109</td>
      <td>0.000190</td>
      <td>0.003204</td>
      <td>-0.002182</td>
      <td>-0.006991</td>
      <td>0.000192</td>
      <td>...</td>
      <td>0.000607</td>
      <td>-0.003694</td>
      <td>-0.002330</td>
      <td>0.002030</td>
      <td>0.000092</td>
      <td>-0.000009</td>
      <td>0.001399</td>
      <td>-0.001088</td>
      <td>1.000000</td>
      <td>0.003141</td>
    </tr>
    <tr>
      <th>var_49</th>
      <td>-0.051718</td>
      <td>-0.005945</td>
      <td>-0.005352</td>
      <td>-0.000372</td>
      <td>-0.002987</td>
      <td>0.001484</td>
      <td>0.008009</td>
      <td>-0.001663</td>
      <td>-0.004983</td>
      <td>0.000810</td>
      <td>...</td>
      <td>0.002123</td>
      <td>0.000379</td>
      <td>-0.000821</td>
      <td>0.002975</td>
      <td>0.000081</td>
      <td>-0.003859</td>
      <td>-0.001607</td>
      <td>-0.000892</td>
      <td>0.003141</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>51 rows × 51 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fd49d7f2-b036-4520-aad4-710c95b91668')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-fd49d7f2-b036-4520-aad4-710c95b91668 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fd49d7f2-b036-4520-aad4-710c95b91668');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Display the correlation matrix

plt.figure(figsize=(20, 20))
sns.heatmap(transaction_df_target_feature.corr(),
            annot=False,      # change to false, to many numbers
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix')
plt.show()

```


    
![png](Project.1-Pass-04_files/Project.1-Pass-04_51_0.png)
    



```python
# Calculate a correlation matrix (features only)
transaction_df_features_only = transaction_df_target_feature[feature_cols]
feature_corr = transaction_df_features_only.corr()

plt.figure(figsize=(20, 12))
sns.heatmap(feature_corr,
            annot=False,
            cmap='coolwarm',  # blue = negative, red = positive
            center=0,         # white/gray = no correlation
            vmin=-1, vmax=1)  # full correlation range
plt.title('Correlation Matrix - Features Only')
plt.show()

#output- correlation between features are near 0
#look independent, 2nd GNB assumption

```


    
![png](Project.1-Pass-04_files/Project.1-Pass-04_52_0.png)
    



```python
# Create two data frames

# no transaction (target= 0)
transaction_df_no_success= transaction_df_target_feature[transaction_df_target_feature['target'] == 0].copy()

# successful transaction (target= 1)
transaction_df_successful= transaction_df_target_feature[transaction_df_target_feature['target'] == 1].copy()


print(type(transaction_df_successful))
print(type(transaction_df_no_success))

print(transaction_df_successful.shape)
print(transaction_df_no_success.shape)

```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    (18040, 51)
    (161960, 51)


## Data Processing

### Single Cross-validation


```python
# Split data from into target and features
target =transaction_df_target['target'].copy()
features= transaction_df_target_feature [feature_cols].copy()


```


```python
# Train test split
x_train, x_test, y_train, y_test = train_test_split( features, target, test_size=0.25, random_state=42)
```


```python
# verify shape of each data set from Train test split
for i in (x_train, x_test,y_train,y_test):
  print(i.shape)
  print(type(i))
```

    (135000, 50)
    <class 'pandas.core.frame.DataFrame'>
    (45000, 50)
    <class 'pandas.core.frame.DataFrame'>
    (135000,)
    <class 'pandas.core.series.Series'>
    (45000,)
    <class 'pandas.core.series.Series'>


### Model


```python
# Create an empty GNB model
modelGNB= GaussianNB()
```

### Fit


```python
# Train ( fit ) the model
modelGNB.fit(x_train,y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GaussianNB</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.naive_bayes.GaussianNB.html">?<span>Documentation for GaussianNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GaussianNB()</pre></div> </div></div></div></div>



### Prediction


```python
# Make prediction using the model
pred_result= modelGNB.predict(x_test)
```

### Performance


```python
# Calculate the accuracy
accuracy=metrics.accuracy_score(y_test,pred_result)
accuracy*100
```




    90.91777777777777




```python
# Display a confusion matrix
mat = metrics.confusion_matrix(y_test,pred_result)
names = ['No transaction', 'Transaction']
sns.heatmap(mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=names,  #names start left to right
    yticklabels=names,  #names start from top to bottom
)

plt.xlabel('Actual (what really happened)')
plt.ylabel('Predicted (the guess)') ;
```


    
![png](Project.1-Pass-04_files/Project.1-Pass-04_67_0.png)
    



```python
#Classification report
print(metrics.classification_report(y_test, pred_result,))



```

                  precision    recall  f1-score   support
    
               0       0.92      0.99      0.95     40426
               1       0.66      0.22      0.33      4574
    
        accuracy                           0.91     45000
       macro avg       0.79      0.60      0.64     45000
    weighted avg       0.89      0.91      0.89     45000
    


### Communication of Results

The target show no transaction (‘0’) or  successful transaction (‘1’).

**Accuracy:** 91%


**Confusion matrix:**

 - 39,915: guessed no transaction, actually no transaction → right
  
 - 3576: guessed no transaction, actually successfull transaction → wrong

 - 511: guessed sucessful transaction, actually no transaction -> wrong

 - 998: guess sucessful transaction, actually transaction ->right
  
**Classification report:**
  
  - Recall for successfull transaction: 0.22
    - The model found 998 out 4574 successfull transaction.

  - Precision for successfull transaction: 0.66
    - The model said successfull transaction 1509 times, right 998 times   

  - 91% is just for the no transaction customer.

**Takeaway:** The model guessed "no transaction" for most customer transaction.


