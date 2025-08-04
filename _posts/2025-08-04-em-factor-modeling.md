---
layout: post
title: "Factor Exposure Modeling in Emerging Markets"
image: "/posts/em_factors.jpg"
tags: [Python, Finance, Data Science]
---

# Project Summary

This project explores how macroeconomic variables influence equity factor exposures in emerging markets. Using data on FX, rates, and equity indexes, we build a basic factor model using PCA and regression techniques.

---

## Data Acquisition

## 📥 01 – Data Acquisition

This notebook pulls EM equity and macroeconomic time series data from Bloomberg using the BQL API.

```python

import pandas as pd
import bql
import os

bq = bql.Service()
date_range = bq.func.range('-3Y', '0D')

em_assets = {
    'Brazil_EWZ': 'EWZ US Equity',
    'India_INDA': 'INDA US Equity',
    'China_FXI': 'FXI US Equity',
    'SouthAfrica_EZA': 'EZA US Equity',
    'Mexico_EWW': 'EWW US Equity',
    'Indonesia_EIDO': 'EIDO US Equity'
}

em_data = {}
for label, ticker in em_assets.items():
    data_item = bq.data.px_last(dates=date_range, fill='prev')
    request = bql.Request(ticker, data_item)
    response = bq.execute(request)
    df = response[0].df()
    px_col = [col for col in df.columns if 'PX_LAST' in col.upper()][0]
    df = df[['DATE', px_col]]
    df.columns = ['date', label]
    df.set_index('date', inplace=True)
    em_data[label] = df

em_df = pd.concat(em_data.values(), axis=1)

macro_assets = {
    'USD_Index': 'DXY Curncy',
    'Oil_Brent': 'CO1 Comdty',
    'US_10Y_Yield': 'USGG10YR Index',
    'Fed_Funds': 'FDTR Index',
    'VIX': 'VIX Index',
    'Copper': 'LMCADY Comdty'
}

macro_data = {}
for label, ticker in macro_assets.items():
    data_item = bq.data.px_last(dates=date_range, fill='prev')
    request = bql.Request(ticker, data_item)
    response = bq.execute(request)
    df = response[0].df()
    px_col = [col for col in df.columns if 'PX_LAST' in col.upper()][0]
    df = df[['DATE', px_col]]
    df.columns = ['date', label]
    df.set_index('date', inplace=True)
    macro_data[label] = df

macro_df = pd.concat(macro_data.values(), axis=1)
combined_df = pd.merge(em_df, macro_df, left_index=True, right_index=True)
combined_df = combined_df.sort_index().dropna()

os.makedirs('../data', exist_ok=True)
combined_df.to_csv('../data/combined_em_macro_data.csv')
combined_df.head()

```

```python

```

---

## Factor Modeling

## 📊 02 – Factor Modeling with PCA and Regression

```python

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/combined_em_macro_data.txt', parse_dates=['date'], index_col='date')
log_returns = np.log(df / df.shift(1)).dropna()

em_columns = [col for col in df.columns if col.startswith(('Brazil', 'India', 'China', 'SouthAfrica', 'Mexico', 'Indonesia'))]
macro_columns = [col for col in df.columns if col not in em_columns]

Y = log_returns[em_columns]
X = log_returns[macro_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(6, 4))
plt.plot(range(1, 4), explained_var, marker='o')
plt.title('PCA Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.show()

betas = {}
r2_scores = {}
for col in Y.columns:
    model = LinearRegression().fit(X_pca, Y[col])
    betas[col] = model.coef_
    r2_scores[col] = model.score(X_pca, Y[col])

beta_df = pd.DataFrame(betas, index=['PC1', 'PC2', 'PC3']).T

plt.figure(figsize=(8, 5))
sns.heatmap(beta_df, annot=True, cmap='coolwarm', center=0)
plt.title('Sensitivity of EM Equities to Macro Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('EM Equity Index')
plt.tight_layout()
plt.show()

sample_col = 'Brazil_EWZ'
model = LinearRegression().fit(X_pca, Y[sample_col])
Y_pred = model.predict(X_pca)

plt.figure(figsize=(10, 4))
plt.plot(Y.index, Y[sample_col], label='Actual', linewidth=1.5)
plt.plot(Y.index, Y_pred, label='Predicted (PCA Model)', linestyle='--')
plt.title(f"{sample_col} Return: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

```

![png](output_1_0.png)

![png](output_1_1.png)

![png](output_1_2.png)

```python
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and process data
df = pd.read_csv('../data/combined_em_macro_data.txt', parse_dates=['date'], index_col='date')
log_returns = np.log(df / df.shift(1)).dropna()

# Separate EM indices and macro variables
em_columns = [col for col in df.columns if col.startswith(('Brazil', 'India', 'China', 'SouthAfrica', 'Mexico', 'Indonesia'))]
macro_columns = [col for col in df.columns if col not in em_columns]

Y = log_returns[em_columns]
X = log_returns[macro_columns]

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

# Plot explained variance
plt.figure(figsize=(6, 4))
plt.plot(range(1, 4), explained_var, marker='o')
plt.title('PCA Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit linear model for each EM index
betas = {}
r2_scores = {}
for col in Y.columns:
    model = LinearRegression().fit(X_pca, Y[col])
    betas[col] = model.coef_
    r2_scores[col] = model.score(X_pca, Y[col])

beta_df = pd.DataFrame(betas, index=['PC1', 'PC2', 'PC3']).T

# Heatmap of betas
plt.figure(figsize=(8, 5))
sns.heatmap(beta_df, annot=True, cmap='coolwarm', center=0)
plt.title('Sensitivity of EM Equities to Macro Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('EM Equity Index')
plt.tight_layout()
plt.show()

# Create plots output directory
plot_dir = '../output/plots'
os.makedirs(plot_dir, exist_ok=True)

# Plot and save actual vs. predicted charts for all EM indices
for col in Y.columns:
    model = LinearRegression().fit(X_pca, Y[col])
    Y_pred = model.predict(X_pca)
    r2 = model.score(X_pca, Y[col])

    plt.figure(figsize=(10, 4))
    plt.plot(Y.index, Y[col], label='Actual', linewidth=1.5)
    plt.plot(Y.index, Y_pred, label='Predicted', linestyle='--')
    plt.title(f'{col} — Actual vs Predicted (R² = {r2:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save each figure
    filename = col.replace(" ", "_").replace("/", "_") + '.png'
    plt.savefig(os.path.join(plot_dir, filename))

local_output_path = '../output/plots'
os.makedirs(local_output_path, exist_ok=True)
# Create a DataFrame for R² scores
r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R² Score'])
r2_df.index.name = 'EM Equity Index'
r2_df.sort_values("R² Score", ascending=False, inplace=True)

# Save the plot
plot_filename = os.path.join(local_output_path, "r2_scores_by_em_index.png")
r2_df.plot(kind='bar', legend=False, color='skyblue', edgecolor='black')
plt.ylabel("R² Score")
plt.title("Model Fit (R²) by EM Equity Index")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y')
plt.savefig(plot_filename)

plot_filename
```

![png](output_2_0.png)

![png](output_2_1.png)

![png](output_2_2.png)

    <Figure size 640x480 with 0 Axes>

![png](output_2_4.png)

    <Figure size 640x480 with 0 Axes>

![png](output_2_6.png)

    <Figure size 640x480 with 0 Axes>

![png](output_2_8.png)

    <Figure size 640x480 with 0 Axes>

![png](output_2_10.png)

    <Figure size 640x480 with 0 Axes>

![png](output_2_12.png)

    '../output/plots/r2_scores_by_em_index.png'




    <Figure size 640x480 with 0 Axes>

![png](output_2_15.png)

```python

```

---

## Visualization and Analysis

# 📈 03 – Visualizations, Rolling Regression & Reusable Functions

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/combined_em_macro_data.txt', parse_dates=['date'], index_col='date')
log_returns = np.log(df / df.shift(1)).dropna()

em_cols = [c for c in df.columns if c.startswith(('Brazil', 'India', 'China', 'SouthAfrica', 'Mexico', 'Indonesia'))]
macro_cols = [c for c in df.columns if c not in em_cols]

Y_all = log_returns[em_cols]
X_all = log_returns[macro_cols]

def rolling_r2_scores(X, Y, window=60, n_components=3):
    results = pd.DataFrame(index=Y.index[window:], columns=Y.columns)
    for col in Y.columns:
        for i in range(window, len(Y)):
            X_window = X.iloc[i - window:i]
            Y_window = Y[col].iloc[i - window:i]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_window)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            model = LinearRegression().fit(X_pca, Y_window)
            results.at[Y_window.index[-1], col] = model.score(X_pca, Y_window)

    return results.astype(float)

rolling_r2 = rolling_r2_scores(X_all, Y_all, window=60)

sample_col = 'Brazil_EWZ'
plt.figure(figsize=(10, 4))
plt.plot(rolling_r2.index, rolling_r2[sample_col])
plt.title(f'Rolling R²: {sample_col} vs Macro Factors (60-day PCA model)')
plt.ylabel('R²')
plt.grid(True)
plt.tight_layout()
plt.show()

```

![png](output_1_0.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulated example to create a rolling_r2 DataFrame
dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
countries = ['Brazil_EWZ', 'India_NIFTY', 'China_CSI300', 'SouthAfrica_JSE', 'Mexico_MXX', 'Indonesia_JKSE']
rolling_r2 = pd.DataFrame(
    np.random.rand(100, len(countries)),
    index=dates,
    columns=countries
)

# Output directory
output_dir = "../output/plots"
os.makedirs(output_dir, exist_ok=True)

# Generate and save one chart per country
for col in rolling_r2.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_r2.index, rolling_r2[col])
    plt.title(f'Rolling R²: {col} vs Macro Factors (60-day PCA model)')
    plt.ylabel('R²')
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    filename = f"rolling_r2_{col.replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

output_dir
```

    '../output/plots'

```python

```

---

## Summary and Insights

# 📊 Project Summary Report

**Project:** Macro Factor Modeling for Emerging Markets  
**Author:** Your Name  
**Date:** YYYY-MM-DD

## 🔍 Project Objective

Brief summary of the problem you're solving and why it matters.

## 🧮 Methods Used

- PCA on standardized macroeconomic time series
- Rolling linear regression to model EM index returns
- Visual R² tracking to evaluate fit quality

## 📈 Key Visuals

Include your best plots below.

```python
from IPython.display import Image, display
display(Image('../output/plots/r2_scores_by_em_index.png'))
```

![png](output_4_0.png)

## 🧠 Interpretation & Insights

Write a few bullets interpreting your results:

- Brazil shows highest macro sensitivity (R² ≈ 0.72)...
- China returns were poorly explained...

## 🛠️ Tools

- Python, pandas, scikit-learn, matplotlib
- Bloomberg BQuant for original data

## ✅ Next Steps or Improvements

- Add Lasso regression for feature selection
- Expand macro variables to include trade balances or PMIs
- Test model stability during crisis periods

```python
# Here are some figure ideas to enhance your project:
# 1. Scree plot of PCA explained variance
# 2. Time series plot of principal component scores
# 3. Rolling window R² for each EM index (line plot)
# 4. Heatmap of factor loadings (PCA components vs. macro variables)
# 5. Residuals plot for regression diagnostics
# 6. Correlation matrix heatmap of macro variables

# Example: Scree plot for PCA explained variance (assuming you have a fitted PCA object `pca`)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.show()
```

```python
import matplotlib.pyplot as plt

# Example: Scree plot for PCA explained variance (assuming you have a fitted PCA object `pca`)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()
```

```python
import seaborn as sns

# 2. Time series plot of principal component scores (assuming you have `pc_scores` DataFrame)
plt.figure(figsize=(10, 4))
for col in pc_scores.columns:
    plt.plot(pc_scores.index, pc_scores[col], label=col)
plt.title('Principal Component Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Rolling window R² for each EM index (assuming you have `rolling_r2` DataFrame)
plt.figure(figsize=(10, 4))
for col in rolling_r2.columns:
    plt.plot(rolling_r2.index, rolling_r2[col], label=col)
plt.title('Rolling Window R² for Each EM Index')
plt.xlabel('Date')
plt.ylabel('R²')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Heatmap of factor loadings (assuming you have `pca.components_` and `macro_var_names`)
plt.figure(figsize=(8, 6))
sns.heatmap(pca.components_, annot=True, cmap='coolwarm',
            yticklabels=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            xticklabels=macro_var_names)
plt.title('PCA Factor Loadings')
plt.xlabel('Macro Variable')
plt.ylabel('Principal Component')
plt.tight_layout()
plt.show()

# 5. Residuals plot for regression diagnostics (assuming you have `residuals` Series)
plt.figure(figsize=(8, 4))
plt.plot(residuals.index, residuals.values)
plt.title('Regression Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()

# 6. Correlation matrix heatmap of macro variables (assuming you have `macro_df`)
plt.figure(figsize=(8, 6))
sns.heatmap(macro_df.corr(), annot=True, cmap='vlag')
plt.title('Correlation Matrix of Macro Variables')
plt.tight_layout()
plt.show()
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 4
          1 import seaborn as sns
          3 # 2. Time series plot of principal component scores (assuming you have `pc_scores` DataFrame)
    ----> 4 plt.figure(figsize=(10, 4))
          5 for col in pc_scores.columns:
          6     plt.plot(pc_scores.index, pc_scores[col], label=col)


    NameError: name 'plt' is not defined

```python

```

```python

```

---

## 📈 Full Notebooks

- [Data Acquisition Notebook](../notebooks/01_data_acquisition.ipynb)
- [Factor Modeling Notebook](../notebooks/02_factor_modeling.ipynb)
- [Visualization and Summary](../notebooks/04_summary_report.ipynb)

---

## Conclusion

This analysis demonstrates how simple data science techniques can be used to decompose macro risk exposures. Future work might incorporate more dynamic modeling or high-frequency indicators.
