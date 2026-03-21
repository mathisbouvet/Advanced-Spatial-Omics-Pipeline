<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Analyse Comparative entre l'intensité des marqueurs et la taille cellulaire
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucun donnée réel


<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Corrélation en cytométrie spatiale</strong><br>
L'analyse de la relation entre l'intensité des marqueurs protéiques et les paramètres morphologiques (comme la taille des cellules) est cruciale pour comprendre le biais technique ou biologique. Dans le cadre de l'imagerie cyclique (MACSima), cette démarche permet de déterminer si l'expression d'une protéine est proportionnelle au volume cellulaire ou si elle caractérise un état phénotypique indépendant de la taille.
</div>
<br>
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>

Ce pipeline vise à quantifier et visualiser l'influence de l'expression de marqueurs spécifiques sur la morphologie cellulaire. Le processus se décompose en trois étapes : l'évaluation statistique, la modélisation prédictive et la comparaison normalisée

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importation des librairies</strong><br>

<details>
  <summary><b>Afficher/masquer le code de configuration</b></summary>

```python
import importlib
import subprocess
import sys

# Dictionnaire des bibliothèques (Clé: nom du module, Valeur: nom du paquet pip)
required_packages = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "statsmodels": "statsmodels",
    "scipy": "scipy"
}

def install_and_import():
    print("🔍 Analyse de l'environnement de travail...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} est déjà prêt.")
        except ImportError:
            print(f"⚠️ {module_name} manque. Installation de {package_name} en cours...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"🚀 {package_name} installé avec succès !")
            except Exception as e:
                print(f"❌ Erreur lors de l'installation de {package_name} : {e}")

# Exécution de la vérification
install_and_import()



# Importations spécifiques
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

print("\n✨ Toutes les bibliothèques de statistiques et de prétraitement sont prêtes !")
```
</details>
</div>



## Objectif

Ce script vise à étudier la relation entre l'intensité de différents marqueurs cellulaires et la taille des cellules. 



## 1 Chargement des données 

Le script a été initialement pensé pour l'observation de 4 marqueurs : CD105, EGF Receptor, Insl3 et CYP17A1. Il peut néamoins convenir à n'importe quel marqueur qui interviendrait dans la taille cellulaire, voir dans d'autre paramètre morphologique de la cellule.

```python
data = pd.read_csv("[Fichier].csv")

markers = ['(marqueur 1) Cell Intensity Average', '(marqueur 2) Cell Intensity Average',
           '(marqueur 3) Cell Intensity Average', '(marqueur 4) Cell Intensity Average']

# Dictionnaire pour renommer les marqueurs
marker_labels = {}

labels_affiches = [marker_labels.get(m, m) for m in markers]
```

## 2 Analyse de Corrélation

Pour chaque marqueur, on calcule la corrélation de **Pearson** avec la taille cellulaire.

#### Définition Mathématique
La corrélation de Pearson est définie par :

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}$$

Où :
* $r \in [-1, 1]$
* **$r > 0$** : relation positive (les variables évoluent dans le même sens).
* **$r < 0$** : relation négative (les variables évoluent en sens opposé).
* **$r = 0$** : absence de relation linéaire.

#### Significativité Statistique
La **p-value** teste l’hypothèse nulle suivante :

$$H_0 : r = 0$$

> **Règle de décision :** Si $p < 0.05$, alors la corrélation est considérée comme **statistiquement significative**.

```python
correlations = []
p_values = []

for marker in markers:
    intensity = data[marker]
    cell_size = data['Cell Size'] #Ou autre caractéristique morphologique
    correlation, p_value = pearsonr(intensity, cell_size)
    correlations.append(correlation)
    p_values.append(p_value)

    print(f'{marker}:\n  Corrélation de Pearson : {correlation:.4f}\n  P-value : {p_value:.4e}\n')

plt.figure(figsize=(9, 7))
bars = plt.bar(range(len(markers)), correlations, color='#003049', edgecolor='black')

plt.axhline(0, color='black', linestyle='--')

plt.title(" ", color='black')
plt.xlabel('Marqueurs', color='black')
plt.ylabel('Corrélation de Pearson', color='black')

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
## 3. Régression linéaire 

Pour les marqueurs présentant une corrélation significative, une **régression linéaire simple** est ajustée afin de quantifier l'influence du marqueur sur la morphologie :

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

#### Paramètres du modèle :
* **$Y$** : Variable dépendante (taille cellulaire).
* **$X$** : Variable indépendante (intensité du marqueur).
* **$\beta_0$** : L'ordonnée à l'origine (valeur de $Y$ quand $X = 0$).
* **$\beta_1$** : Le coefficient de régression, représentant l'**effet du marqueur**. Une unité d'augmentation de $X$ entraîne une variation de $\beta_1$ unités de $Y$.
* **$\varepsilon$** : Le terme d'erreur (résidus).

> **Interprétation :** Le coefficient $\beta_1$ permet de déterminer si le marqueur est un prédicteur fort de la taille cellulaire.

```python
data = pd.read_csv("[Fichier].csv")

# Choix des marqueurs
markers = ['(marqueur 1)', '(marqueur 2)']

# Dictionnaire
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
        f'Influence de {marker_labels[marker]} sur la taille de la cellule\nCorrélation : {correlation:.2f}',
        color='black', fontsize=12
    )
    axes[i].set_xlabel('Intensité du marqueur', color='black')
    axes[i].set_ylabel('Taille de la cellule', color='black')

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
## 4. Régression sur données standardisées

Une calcule une nouvelle régression sur les données standardisées

```python
data = pd.read_csv("[Fichier].csv")

markers = ['(marqueur 1)', '(marqueur 2)']
marker_labels = {}
colors = ['#6AB8E6', '#E67E22']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

plt.figure(figsize=(10, 6))
plt.title("Influence standardisée des marqueurs sur la taille des cellules", fontsize=14)

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

plt.xlabel("Intensité standardisée du marqueur")
plt.ylabel("Taille standardisée de la cellule")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig("influence_marker_same_graph_standardized.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```

## 5. Piste d'amélioration

Actuellement on suppose que 1 marqueurs = 1 taille alors que biologiquement on sait que la taille peut être la combinaison de plusieurs marqueurs. On pourrait utilser une régression multiple

```python
X = data[markers]
y = data['Cell Size']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
```

On utilise que Pearson qui est uniquement linéaire. On pourrait utiliser Spearman qui est plus robuste 

```python
from scipy.stats import spearmanr

corr, p = spearmanr(intensity, cell_size)
```