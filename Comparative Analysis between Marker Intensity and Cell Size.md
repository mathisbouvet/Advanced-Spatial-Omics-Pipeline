<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Comparative Analysis between Marker Intensity and Cell Size
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  By <strong>Mathis BOUVET</strong> — Biologist specializing in Reproduction and Development
  <br>
  <span style="font-size: 0.8em; font-style: italic;">March 2026</span>
</div>
<br>

> **Important note**
> : This document contains no real data


<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Correlation in spatial cytometry</strong><br>
The analysis of the relationship between protein marker intensity and morphological parameters (such as cell size) is crucial for understanding technical or biological bias. In the context of cyclic imaging (MACSima), this approach makes it possible to determine whether protein expression is proportional to cell volume or whether it reflects a phenotypic state independent of size.
</div>
<br>
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objective
</h2>

This pipeline aims to quantify and visualize the influence of specific marker expression on cell morphology. The process is divided into three steps: statistical evaluation, predictive modeling, and normalized comparison.

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importing libraries</strong><br>

<details>
  <summary><b>Show/hide configuration code</b></summary>

```python
import importlib
import subprocess
import sys

# Library Dictionary
required_packages = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "statsmodels": "statsmodels",
    "scipy": "scipy"
}

def install_and_import():
    print("Analysis of the work environment...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} is already ready.")
        except ImportError:
            print(f"⚠️ {module_name} lack. Installation of {package_name} en cours...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"{package_name} installed successfully !")
            except Exception as e:
                print(f"❌ Error during installation of {package_name} : {e}")

install_and_import()

# Import
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

print("\n✨ All the libraries are ready !")
```
</details>
</div>



## Objective

This script aims to study the relationship between the intensity of different cellular markers and cell size.



## 1 Loading data

The script was initially designed to analyze four markers: CD105, EGF Receptor, Insl3, and CYP17A1. However, it can be applied to any marker involved in cell size or other cellular morphological parameters.

```python
data = pd.read_csv("[Fill].csv")

markers = ['(marker 1) Cell Intensity Average', '(marker 2) Cell Intensity Average',
           '(marker 3) Cell Intensity Average', '(marker 4) Cell Intensity Average']

marker_labels = {}

labels_affiches = [marker_labels.get(m, m) for m in markers]
```

## 2 Correlation Analysis

For each marker, we calculate the **Pearson** correlation with cell size.

#### Mathematical definition
Pearson's correlation is defined by:

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}$$

* $r \in [-1, 1]$
* **$r > 0$** : positive relationship (the variables evolve in the same direction).
* **$r < 0$** : negative relationship (the variables evolve in opposite directions).
* **$r = 0$** : absence of a linear relationship.

#### Significativité Statistique
The **p-value** tests the following null hypothesis:

$$H_0 : r = 0$$

> **Decision rule :** If $p < 0.05$, then the correlation is considered **statistically significant**.

```python
correlations = []
p_values = []

for marker in markers:
    intensity = data[marker]
    cell_size = data['Cell Size'] # Or other morphological characteristic
    correlation, p_value = pearsonr(intensity, cell_size)
    correlations.append(correlation)
    p_values.append(p_value)

    print(f'{marker}:\n  Pearson correlation : {correlation:.4f}\n  P-value : {p_value:.4e}\n')

plt.figure(figsize=(9, 7))
bars = plt.bar(range(len(markers)), correlations, color='#003049', edgecolor='black')

plt.axhline(0, color='black', linestyle='--')

plt.title(" ", color='black')
plt.xlabel('markers', color='black')
plt.ylabel('Pearson correlation', color='black')

plt.xticks(ticks=range(len(markers)), labels=labels_affiches, rotation=45, ha='right', color='black')
plt.yticks(color='black')
plt.ylim(-0.4, 0.4)

annotation_height = 0.42 

for bar, p_val in zip(bars, p_values):
    label = f'p={p_val:.4f}'
    if p_val < 0.05:
        label += ' *'
    height = bar.get_height()
    offset = 0.02 if height >= 0 else -0.06
    plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
             label, ha='center', va='bottom', color='black', fontsize=9)

plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
```
## 3. Linear regression

For markers showing a significant correlation, a **simple linear regression** is fitted to quantify the influence of the marker on morphology:

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

#### Model parameters :
* **$Y$** : Dependent variable (cell size).
* **$X$** : Independent variable (marker intensity).
* **$\beta_0$** : The y-intercept (value of $Y$ when $X = 0$).
* **$\beta_1$** : The regression coefficient represents the **effect of the marker**. One unit increase in $X$ results in a change of $\beta_1$ units in $Y$.
* **$\varepsilon$** : The term error (residues).

> **Interpretation :** The coefficient $\beta_1$ allows us to determine if the marker is a strong predictor of cell size.

```python
data = pd.read_csv("[Fill].csv")

# Marker selection
markers = ['(marker 1)', '(marker 2)']

# Dictionary
marker_labels = {}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), facecolor='white')

for i, marker in enumerate(markers):
    X = data[marker]
    y = data['Cell Size']
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    y_pred = model.predict(X_with_const)
    correlation, _ = pearsonr(X, y)

    axes[i].scatter(X, y, alpha=0.6, color='#6AB8E6', edgecolors='black', label='Données')
    axes[i].plot(X, y_pred, color='#9B59B6', label='Régression', linewidth=2)

    axes[i].set_title(
        f'Influence of {marker_labels[marker]} on cell size\nCorrelation : {correlation:.2f}',
        color='black', fontsize=12
    )
    axes[i].set_xlabel('Marker intensity', color='black')
    axes[i].set_ylabel('Cell size', color='black')

    axes[i].tick_params(axis='x', colors='black')
    axes[i].tick_params(axis='y', colors='black')
    axes[i].legend()
    axes[i].grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    axes[i].axhline(0, color='black', linewidth=0.8)
    axes[i].axvline(0, color='black', linewidth=0.8)
    axes[i].set_facecolor('white')

plt.tight_layout()
plt.show()
```
## 4. Regression on standardized data

A new regression is calculated on the standardized data.

```python
data = pd.read_csv("[Fill].csv")

markers = ['(marker 1)', '(marker 2)']
marker_labels = {}
colors = ['#6AB8E6', '#E67E22']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

plt.figure(figsize=(10, 6))
plt.title("Standardized influence of markers on cell size", fontsize=14)

for i, marker in enumerate(markers):
    X = data[marker].values.reshape(-1, 1)
    y = data['Cell Size'].values.reshape(-1, 1)

    X_std = scaler_X.fit_transform(X).flatten()
    y_std = scaler_y.fit_transform(y).flatten()

    X_with_const = sm.add_constant(X_std)
    model = sm.OLS(y_std, X_with_const).fit()
    y_pred = model.predict(X_with_const)
    correlation, _ = pearsonr(X_std, y_std)

    plt.scatter(X_std, y_std, alpha=0.6, color=colors[i], edgecolors='black',
                label=f'{marker_labels[marker]} (r = {correlation:.2f})')
    plt.plot(X_std, y_pred, color=colors[i], linewidth=2)

plt.xlabel("Standardized marker intensity")
plt.ylabel("Standardized cell size")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.show()
```

## 5. Area for improvement

Currently, we assume that one marker corresponds to one size, whereas biologically we know that size can result from the combination of multiple markers. A multiple regression approach could be used.

```python
X = data[markers]
y = data['Cell Size']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
```

We only use Pearson, which is purely linear. We could use Spearman, which is more robust.

```python
from scipy.stats import spearmanr

corr, p = spearmanr(intensity, cell_size)
```