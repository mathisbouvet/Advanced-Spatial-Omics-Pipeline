<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Protocole de Contrôle Qualité et d'Optimisation de la Segmentation Tissulaire
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucune donnée réel


<span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Issu d'une coupe tissulaire</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Code Python</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">MACSima</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Segmentation</span>

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>La segmentation cellulaire</strong><br>
  Avant d'entamer une analyse spatiale, il est crucial d'obtenir une segmentation fidèle à la réalité biologique du tissu étudié. Si la segmentation manuelle demeure la méthode de référence en termes de précision, les contraintes temporelles imposent le recours à des outils automatisés. Des solutions telles que Cellpose, StarDist ou QuPath offrent désormais des segmentations de haute précision grâce au Deep Learning. Toutefois, ces outils se heurtent à certaines limites, notamment le risque de sur-apprentissage (overfitting) ou l'incapacité à généraliser face à la diversité des architectures tissulaires. L'exigence de segmentation diffère radicalement entre une coupe de testicule et une coupe d'ovaire, par exemple. Dès lors, il devient indispensable de développer des protocoles de validation rigoureux, où la segmentation automatique est confrontée à une "vérité terrain" (ground truth) établie manuellement, afin d'en quantifier la fiabilité.
</div>
<br>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>
Ce document détaille les étapes de l'analyse de la segmentation cellulaire, s'appuyant sur des masques de régions d'intérêt (ROI) issus d'une segmentation manuelle, afin de les confronter aux résultats d'une segmentation automatisée

<br>

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importation des librairies</strong><br>

<details>
  <summary><b>Afficher/masquer le code de configuration</b></summary>

```python
import importlib
import subprocess
import sys

# Dictionnaire des bibliothèques nécessaires
required_packages = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "skimage": "scikit-image",
    "read_roi": "read-roi",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "seaborn": "seaborn"
}
def install_and_import():
    print("Analyse de l'environnement nlp_env...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} est déjà prêt.")
        except ImportError:
            print(f"⚠️ {module_name} manque. Installation de {package_name} en cours...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"🚀 {package_name} installé avec succès !")

install_and_import()

# Importation des bibliothèques après installation
import numpy as np
import cv2
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

from skimage.draw import polygon, polygon_perimeter
from read_roi import read_roi_file
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

print("✨ Tous les modules sont importés et prêts à l'emploi !")
```
</details>
</div>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  1. Création des masques à partir des ROIs
</h2>

### 1.a. Création des ROIs via Fiji et importation dans l'environnement Python


Une image DAPI de référence est extraite du système MACSima, puis exportée au format `.tif` pour être traitée sous Fiji. Des régions d'intérêt (ROI) y sont générées afin de servir de base à la création des masques de segmentation. Une fois ces ROI créées, l'image de référence ainsi que le dossier contenant leurs coordonnées sont importés dans l'environnement Python pour la suite de l'analyse.

> **Image DAPI**
> : En format `tiff` exporté via le MACSIMA

> **SetROIs**
> : En format `zip` créer et exporté à partir de Fiji


```python
# Charger l’image TIFF
image = cv2.imread("[image.tiff]")

# Dossier temporaire pour extraire les ROIs
extract_folder = "roi_dezip"

# Extraire les fichiers du ZIP
with zipfile.ZipFile("[SetROIs.zip]", "r") as zip_ref:
    zip_ref.extractall(extract_folder)
roi_files = [f for f in os.listdir(extract_folder) if f.endswith(".roi")]
all_rois = {}
for roi_file in roi_files:
    roi_path = os.path.join(extract_folder, roi_file)
    roi_data = read_roi_file(roi_path)
    all_rois.update(roi_data)
print(f"Nombre de ROIs chargés : {len(all_rois)}")
```
Attention à bien vérifier le nombre de ROIs importé 


### 1. b Génération des masques 


Dans cette analyse, plusieurs types de masques sont générés pour tester le mode de segmentation de MACSiQView. Le code ainsi exécuté permet d'obtenir 4 types de masques différents 

```python
#ROIs colorées + fond noir
masked_image = np.zeros_like(image)
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    color = generate_random_color()
    masked_image[rr, cc] = color
cv2.imwrite("mask_1.tif", masked_image)

#ROIs colorées (2) + fond noir
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
masked_image = np.zeros_like(image)
roi_color_idx = 0 
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    color = colors[roi_color_idx]
    masked_image[rr, cc] = color
    roi_color_idx = (roi_color_idx + 1) % len(colors)
cv2.imwrite("mask_2.tif", masked_image)

#ROIs en niveau de gris
gray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    gray_mask[rr, cc] = np.clip(gray_mask[rr, cc] + 50, 0, 255)
cv2.imwrite("mask_3.tif", gray_mask)

#ROIs en niveau de gris + contours
gray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for roi_name, roi in roi_data.items():
    x = np.array(roi["x"])
    y = np.array(roi["y"])
    rr, cc = polygon(y, x, shape=image.shape[:2])
    gray_mask[rr, cc] = np.clip(gray_mask[rr, cc] + 50, 0, 255)
    rr_perim, cc_perim = polygon_perimeter(y, x, shape=image.shape[:2])
    gray_mask[rr_perim, cc_perim] = 0
cv2.imwrite("mask_4.tif", gray_mask)
```
<br>

### 1.c Segmentation MACSiQView sur les masques générés par Python et comparaison entre elle


Le logiciel MACSiQView propose différents algorithmes de segmentation. L’objectif de cette étude est d'évaluer les quatre modalités disponibles (Import mask, Import label, Single Cell et Tissue) afin d'identifier celle offrant la segmentation la plus fidèle aux structures biologiques. Une fois la segmentation validée, l’onglet "Feature Table" permet de sélectionner les descripteurs à intégrer dans l’analyse. Cette étape assure l'exportation exclusive des données morphologiques cellulaires, indépendamment des intensités de fluorescence. Le fichier ainsi généré, au format `.csv`, est ensuite importé dans un environnement Python. Parmi l’ensemble des variables extraites, les paramètres suivants ont été retenus : Area, Perimeter, Centroid X et Y, Feret et Mean Intensity .

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
2 Comparaison des différentes segmentations 
</h2>

### 2.a  Calcule des comparaisons



Tout d'abord, on importe les segmentations automatiques `df_auto` et la segmentation manuel `df_manual`. On garde seulement les points de comparaison suivant : Area, Perimeter, Centroid X et Y, Feret et Mean intensity. 

On applique une normalisation par `MinMaxScaler()` et on construit un dataframe de comparaison `df_comparison`. Le choix du `MinMaxScaler` repose sur l'hypothèse d'un nettoyage préalable des données sous MACSiQView, garantissant l'absence d'outliers extrêmes et permettant une comparaison directe sur une échelle normalisée [0, 1] plus lisible pour l'analyse morphologique.

```python
# Fichier manuel (issu de Fiji)
df_manual = pd.read_csv("[Segmentation_manuelle].csv")

# Fichiers automatiques (issu de MACSiQView)
df_auto1 = pd.read_csv("[Mask_3_Single_Cell].csv")
df_auto2 = pd.read_csv("[Mask_3_Tissue.csv]")
df_auto3 = pd.read_csv("[Mask_4_Import_mask].csv")
df_auto4 = pd.read_csv("[Mask_4_Single_Cell].csv")
df_auto5 = pd.read_csv("[Mask_4_Tissue].csv")

# Renommer les colonnes
df_manual = df_manual.rename(columns={
    'Area': 'area',
    'Perim.': 'perimeter',
    'XM': 'centroid_x',
    'YM': 'centroid_y',
    'Feret': 'feret',
    'Mean': 'mean_intensity'
})

# Conserver uniquement les colonnes spécifiées
columns_to_keep = ['area', 'perimeter', 'centroid_x', 'centroid_y', 'feret', 'mean_intensity']
df_manual = df_manual[columns_to_keep]
print("Avant normalisation :")
print(df_manual.head(), "\n")

# Adaptation des colonnes des fichiers de segmentation automatique
def rename_auto(df):
    return df.rename(columns={
        'Nucleus Size': 'area',
        'Nucleus Contour Length': 'perimeter',
        'Nuc X': 'centroid_x',
        'Nuc Y': 'centroid_y',
        'Nucleus Feret Diameter Max': 'feret',
        'Nucleus DNA Mean': 'mean_intensity'
    })
df_auto1 = rename_auto(df_auto1)
df_auto2 = rename_auto(df_auto2)
df_auto3 = rename_auto(df_auto3)
df_auto4 = rename_auto(df_auto4)
df_auto5 = rename_auto(df_auto5)

scaler = MinMaxScaler()

# Normaliser les colonnes numériques
df_manual[columns_to_keep] = scaler.fit_transform(df_manual[columns_to_keep])
df_auto1[columns_to_keep] = scaler.fit_transform(df_auto1[columns_to_keep])
df_auto2[columns_to_keep] = scaler.fit_transform(df_auto2[columns_to_keep])
df_auto3[columns_to_keep] = scaler.fit_transform(df_auto3[columns_to_keep])
df_auto4[columns_to_keep] = scaler.fit_transform(df_auto4[columns_to_keep])
df_auto5[columns_to_keep] = scaler.fit_transform(df_auto5[columns_to_keep])

print("Après normalisation :")
print(df_manual.head())
print(df_auto1.head())

df_comparison = df_manual[['area', 'perimeter', 'centroid_x', 'centroid_y', 'feret', 'mean_intensity']].copy()

# Ajout des segmentations automatiques
for i, df_auto in enumerate([df_auto1, df_auto2, df_auto3, df_auto4, df_auto5], start=1):
    suffix = f'_auto{i}'
    df_comparison[f'area{suffix}'] = df_auto['area']
    df_comparison[f'perimeter{suffix}'] = df_auto['perimeter']
    df_comparison[f'centroid_x{suffix}'] = df_auto['centroid_x']
    df_comparison[f'centroid_y{suffix}'] = df_auto['centroid_y']
    df_comparison[f'feret{suffix}'] = df_auto['feret']
    df_comparison[f'mean_intensity{suffix}'] = df_auto['mean_intensity']
```


### 2.b Comparaison statistique de la segmentation la plus proche (Kolmogorov-Smirnov)



On utilise un test de Kolmogorov-Smirnov à deux échantillons. Pour simplifier les démarches, on récupères uniquement la statistique KS, on la stock dans un dataframe spécifique `distance_df`et on ajoute la distance moyenne KS sur les 6 paramètres

En analysant les paramètres, le script informe directement la segmentation la plus proche


```python
# Correspondance pour les nouveaux noms
legend_mapping = {
    'Auto1': 'Mask 3 en Single Cell',
    'Auto2': 'Mask 3 en Tissue',
    'Auto3': 'Mask 4 en Import Mask',
    'Auto4': 'Mask 4 en Single Cell',
    'Auto5': 'Mask 4 en Tissue'
}

# Calcule ses distances de Kolmogorov-Smirnov pour chaque paramètre
distances = {}
for column in columns_to_keep:
    distances[column] = []
    for df_auto in [df_auto1, df_auto2, df_auto3, df_auto4, df_auto5]:
        ks_stat, _ = ks_2samp(df_manual[column], df_auto[column])
        distances[column].append(ks_stat)

distances_df = pd.DataFrame(distances, index=['Auto1', 'Auto2', 'Auto3', 'Auto4', 'Auto5'])
distances_df['Mean Distance'] = distances_df.mean(axis=1)
print("Distances moyennes de Kolmogorov-Smirnov pour chaque df_auto :")
print(distances_df['Mean Distance'])

closest_auto = distances_df['Mean Distance'].idxmin()
print(f"\nLa df_auto la plus proche de df_manual est : {legend_mapping[closest_auto]}")
```

### 2.c Visualisation


On réalise également les barplot des distances moyennes

```python
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
ax = distances_df['Mean Distance'].plot(kind='bar', color='#E9EDC9', edgecolor='white')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='white')
plt.title('Distances Moyennes de Kolmogorov-Smirnov par Segmentation', color='white')
plt.ylabel('Distance Moyenne', color='white')
plt.xlabel('Segmentations', color='white')
plt.xticks(ticks=range(len(legend_mapping)), labels=[legend_mapping[item] for item in distances_df.index], rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.6, color='white')
ax.set_facecolor('black')
ax.figure.set_facecolor('black')
plt.savefig('distances_moyennes_ks.png', format='png', facecolor='black')
plt.show()
```
On visualise également les courbes de densité (KDE) pour comparer la forme des distributions
```python
plt.style.use('default')
params = ['area', 'perimeter', 'feret', 'mean_intensity']

# mapping adapté aux clés
legend_mapping = {
    'auto1': 'Mask 3 en Single Cell',
    'auto2': 'Mask 3 en Tissue',
    'auto3': 'Mask 4 en Import Mask',
    'auto4': 'Mask 4 en Single Cell',
    'auto5': 'Mask 4 en Tissue',
    'manuel': 'Référence Manuelle'
}

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for idx, param in enumerate(params):
    ax = axes[idx]
    ax.set_facecolor('white')
    for i in range(1, 6):
        key = f'auto{i}'
        colname = f'{param}_{key}'
        if colname in df_comparison.columns:
            sns.kdeplot(df_comparison[colname], label=legend_mapping[key], linewidth=2, ax=ax)
        else:
            print(f"Colonne manquante : {colname}")

# Vérification si la colonne manuelle existe
    if param in df_comparison.columns:
        sns.kdeplot(df_comparison[param], label=legend_mapping['manuel'], color='cyan', linestyle='--', linewidth=2, ax=ax)
    else:
        print(f"Colonne manuelle manquante : {param}")
    ax.set_title(f'Distribution Comparée - {param}')
    ax.set_xlabel('Valeur Normalisée')
    ax.set_ylabel('Densité')
    ax.grid(True, linestyle='--', alpha=0.6)

# Légende
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Segmentation', loc='upper center', ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('distributions_comparées.png', format='png', facecolor='white')
plt.show()
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
3. Bonne segmentation ?
</h2>

### 3.a Calcul de comparaison entre la segmentation manuelle et automatique



Cette étude a pour objet l’évaluation comparative d’une segmentation automatisée réalisée sous MACSiQ View par rapport à une méthodologie de référence établie à partir de masques de régions d'intérêt (ROI) générés manuellement. Les données issues de la segmentation manuelle, déjà compilées, constituent le jeu de données de référence `df_ref`. La présente analyse vise à appliquer une procédure identique à une image DAPI issue d'une immunofluorescence MACSima, dont les résultats sont consignés dans le jeu de données `df_test`. En assurant une stricte uniformité des paramètres d'exportation, cette approche permet de quantifier la précision de la segmentation automatisée, d'en valider la pertinence scientifique et de formuler des hypothèses d'optimisation pour les protocoles de traitement d'image ultérieurs.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
Les étapes de vérification intégrées au script, bien qu'elles en augmentent la complexité, permettent de s'assurer que les colonnes sont conformes aux paramètres préalablement sélectionnés.
</div>

<br>

Une fois les colonnes triés et les données normalisés (c'est obligatoire). On utilise Isolation Forest qui sera entraîné uniquement sur la référence `X_ref_scaled` (car nous avons segmenté manuellement qu'une seule image). On suppose une contamination de 10%

Pour vérifier la sensibilité au paramètre de contamination, on fait tourner le modèle d'isolation sur différente valeur de contamination. Pour chaque valeur, on regarde quel pourcentage de cellules test est classé "OK"

```python 
# Chargement des fichiers et sélection des colonnes
df_ref = pd.read_csv("Ref")
df_test = pd.read_csv("Test.csv")
df_ref.columns = df_ref.columns.str.strip()
df_test.columns = df_test.columns.str.strip()
features_to_use = [
    "Cell Bbox X Size", "Cell Bbox Y Size", "Cell Shape Circle Like", "Cell Shape Ellipse Like", 
    "Cell Shape Elongation", "Cell Shape Square Like", "Cell Shape Triangle Like", 
    "Cell Size", "Nucleus Size", "Nucleus Roundness", "Nucleus Convexity", "Cell Convexity",
    "Quality Cell In-Focus", "Quality Nuclear Segmentation"
]

# Vérification s'il exsite des colonnes manquantes
missing = [col for col in features_to_use if col not in df_ref.columns or col not in df_test.columns]
if missing:
    raise ValueError(f"❌ Colonnes manquantes dans les fichiers : {missing}")

X_ref = df_ref[features_to_use].dropna()
X_test = df_test[features_to_use].dropna()
df_test_clean = df_test.loc[X_test.index].copy()
scaler = StandardScaler()
X_ref_scaled = scaler.fit_transform(X_ref)
X_test_scaled = scaler.transform(X_test)
```
```python
best_contamination = 0.10
model = IsolationForest(contamination=best_contamination, random_state=42)
model.fit(X_ref_scaled)
preds = model.predict(X_test_scaled)
df_test_clean["Segmentation_OK"] = preds
nb_total = len(preds)
nb_valides = (preds == 1).sum()
pourcentage = nb_valides / nb_total * 100

print(f"\n✅ Résultat avec contamination={best_contamination:.2f} :")
print(f"{nb_valides} / {nb_total} cellules considérées comme bien segmentées ({pourcentage:.2f}%)")
```


```python
contamination_values = [0.01, 0.05, 0.10, 0.15, 0.20]
ok_percentages = []

for c in contamination_values:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(X_ref_scaled)
    preds = model.predict(X_test_scaled)
    ok_percent = (preds == 1).sum() / len(preds) * 100
    ok_percentages.append(ok_percent)

plt.plot(contamination_values, ok_percentages, marker='o')
plt.xlabel("Taux de contamination")
plt.ylabel("% de cellules bien segmentées")
plt.title("Effet de 'contamination' sur la détection des bonnes segmentations")
plt.grid(True)
plt.tight_layout()
plt.show()
df_test_clean.to_csv("segmentation_test_annotated.csv", index=False)
print("💾 Résultat sauvegardé dans : segmentation_test_annotated.csv")
```


### 3.b Proposition d'amélioration de paramètre



Si le modèle Isolation Forest identifie des vecteurs de paramètres comme étant aberrants ou non conformes à une segmentation optimale, il devient alors possible d'extraire des indicateurs d'optimisation pour les réglages de segmentation. À cet effet, un dictionnaire de correspondance est établi afin de lier chaque variable mesurée à un paramètre spécifique de MACSiQ View, permettant ainsi de déterminer l'ajustement requis (incrémentation ou décrémentation du paramètre).Le test statistique de Mann-Whitney U est employé pour comparer les distributions des populations conformes (OK) et non conformes (KO). Si une différence statistiquement significative est observée ($p < 0,01$), l'analyse de la position des médianes (ou moyennes) permet de définir une polarité de l'écart ($KO < OK$ ou $KO > OK$). Cette divergence est ensuite traduite en recommandations opérationnelles pour MACSiQ View, accompagnée du calcul du pourcentage de variation relative entre les deux groupes. »

```python
param_map = {
    "Nucleus Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox X Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox Y Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Nucleus Roundness": ("Séparation / Smoothing", "↑ séparation", "↓ séparation ou ↓ smoothing"),
    "Nucleus Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Shape Ellipse Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Circle Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Elongation": ("Contours / Séparation", "↓ séparation", "↑ séparation"),
    "Cell Shape Square Like": ("Contours", "-", "-"),
    "Cell Shape Triangle Like": ("Contours", "-", "-"),
    "Quality Cell In-Focus": ("Qualité image (acquisition)", "-", "-"),
    "Quality Nuclear Segmentation": ("Sensibilité / Smoothing", "↑ sensibilité", "↓ sensibilité"),
}
df = df_test_clean.copy()
df = df.dropna(subset=features_to_use + ['Segmentation_OK'])
df['Segmentation_Label'] = df['Segmentation_OK'].map({1: 'OK', -1: 'KO'})
summary = []
for feature in features_to_use:
    ok_vals = df[df['Segmentation_Label'] == 'OK'][feature]
    ko_vals = df[df['Segmentation_Label'] == 'KO'][feature]
    if len(ok_vals) < 10 or len(ko_vals) < 10:
        continue
    stat, p = mannwhitneyu(ok_vals, ko_vals, alternative='two-sided')
    mean_ok = ok_vals.mean()
    mean_ko = ko_vals.mean()
    direction = "-"
    suggestion = "-"
    percent_change = "-"
    param = param_map.get(feature, ("-", "-", "-"))[0]

    if p < 0.01:
        if mean_ko < mean_ok:
            direction = "KO < OK"
            suggestion = param_map[feature][1]
            percent_change = f"+{round((1 - mean_ko / mean_ok) * 100, 1)} %"
        elif mean_ko > mean_ok:
            direction = "KO > OK"
            suggestion = param_map[feature][2]
            percent_change = f"+{round((mean_ko / mean_ok - 1) * 100, 1)} %"

    summary.append({
        "Variable": feature,
        "Moyenne OK": round(mean_ok, 2),
        "Moyenne KO": round(mean_ko, 2),
        "Différence": direction,
        "Paramètre MACSiQ lié": param,
        "Suggestion d’ajustement": suggestion,
        "% de changement indicatif": percent_change,
        "p-value": round(p, 4)
    })
summary_df = pd.DataFrame(summary).sort_values("p-value")

# Affichage console
print("\n Résumé des paramètres MACSiQ à ajuster :\n")
print(summary_df.to_string(index=False))

# Export CSV
summary_df.to_csv("macsiq_param_suggestions.csv", index=False)
print("\n💾 Résumé exporté dans : macsiq_param_suggestions.csv")
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
4. Amélioration possible et limite 
</h2>



### 4.a Limite sur le modèle de comparaison
Bien que la comparaison par test de Kolmogorov-Smirnov et la visualisation des densités (KDE) permettent de valider globalement la fidélité d'une méthode de segmentation, ces approches présentent une limite intrinsèque : elles traitent les paramètres de manière isolée (analyse univariée). En réalité, une cellule "aberrante" n'est pas forcément détectable sur un seul critère (ex: une surface normale), mais par la combinaison incohérente de plusieurs facteurs (ex: une surface normale associée à une circularité extrêmement faible et une intensité hétérogène). L'œil humain repère ces anomalies instinctivement, mais une analyse statistique classique peut laisser passer ces artefacts, ce qui biaise les résultats finaux de l'analyse spatiale.

Pour pallier cette limite, on pourrait utiliser l'Isolation Forest. Contrairement aux méthodes qui cherchent à définir un modèle de "cellule parfaite", cet algorithme de Machine Learning identifie les anomalies par leur isolation. Dans un espace mathématique où chaque paramètre morphologique est une dimension, l'algorithme segmente aléatoirement les données : les cellules normales, très denses et similaires, demandent de nombreuses étapes pour être isolées, tandis que les erreurs de segmentation (artefacts, doubles, débris) sont isolées très rapidement.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
On utilise la fonction `sklearn.ensemble.IsolationForest`qu'on utilise déjà partiellement mais pour répondre à la limite univariée. La, on l'utiliserait comme filtre en amont. Ensuite `model.predict(X_test_scaled)` obtient des labels -1 ou 1. Cette étiquettage sera pris en compte dans les graphiques KDE et les test KS
</div>

### 4.b Evolution du modèle vers un DeepLearning

L'étape ultime de ce protocole consiste à dépasser la validation sur une image unique pour construire un modèle de référence robuste. Actuellement, notre analyse s'appuie sur une image segmentée manuellement. Toutefois, la diversité des tissus nécessite une base de données plus large. En compilant les segmentations manuelles de plusieurs coupes tissulaires, nous pouvons constituer un Dataset d'entraînement massif. Il servira à entraîner un modèle de DeepLearning spécifiques à nos conditions d'acquisition. Le workflow devient alors cyclique. Chaque nouvelle image validée par l'Isolation Forest et corrigée par l'humain. Plus le modèle voit de structures cellulaires variées, plus sa capacité de généralisation augmente, réduisant ainsi le besoin de correction manuelle. À terme, ce modèle auto-apprenant permet d'uniformiser la qualité de segmentation sur l'ensemble des projets de l'unité, garantissant une reproductibilité totale, indépendamment de l'opérateur ou de la variabilité biologique du tissu.

<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
On utilise plus des CSV mais des modèles entraîné sur des pixels. Des fonctions <strong>cellpose.models.CellposeModel</strong> ou <strong>standist.models.StarDist2D</strong> peuvent être utilisé. Ça nécessite des ROIs transformé en masque binaire
</div>
<br>
Pour aller plus loin, il sera nécessaire de prendre en compte dans notre modèle la morphologie d'un tissu. De cette manière, nous pouvons pondéré notre analyse en forçant l'algorithme à accorder un poids statistique supérieur aux images de référence issues du même organe. À chaque nouvelle analyse d'un tissu inconnu, les segmentations validées manuellement sont étiquetées et intégrées à la bibliothèque. Le système s'enrichit ainsi organiquement, comblant ses propres lacunes au fil des projets de recherche.

<br>
<div style="border: 1px solid #fff200; border-radius: 10px; padding: 20px; background-color: rgba(234, 255, 0, 0.1); color: #000000;">
Une fois la construction du modèle de DeepLearning, il faut simplement utiliser des dictionnaires de modèles via des fonctions comme <strong>joblib.dump</strong>
</div>
