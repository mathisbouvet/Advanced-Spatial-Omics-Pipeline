<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Analyse Comparative de Clustering sur Données de Protéomique Spatiale (MACSima)
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucun donnée réel


<span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Issu d'une coupe tissulaire</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Code Python</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">MACSima</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Clustering</span>

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Le clustering</strong><br>
Le clustering est une méthode d'apprentissage non supervisé visant à regrouper des entités (ici, des cellules) présentant des profils d'expression protéique similaires. Dans le cadre de l'imagerie cyclique, cette technique permet d'identifier des populations cellulaires de manière agnostique, sans a priori biologique, afin de révéler l'hétérogénéité phénotypique au sein d'un tissu. L'enjeu est de partitionner l'espace multidimensionnel des marqueurs pour définir des signatures cellulaires distinctes.
</div>
<br>
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>


L'objectif de ce pipeline est d'automatiser le choix de l'algorithme de partitionnement le plus performant pour un jeu de données donné. Le processus ce décompose en trois étapes : validation de la structure, optimisation du k et benchmark de domdèles

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importation des librairies</strong><br>

<details>
  <summary><b>Afficher/masquer le code de configuration</b></summary>

```python
import importlib
import subprocess
import sys

# Dictionnaire des bibliothèques
required_packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "minisom": "minisom",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy"
}

def install_and_import():
    print("Analyse de l'environnement de travail...")
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

install_and_import()


# Importations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from minisom import MiniSom  

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

warnings.filterwarnings('ignore')

print("\n✨ Toutes les bibliothèques de clustering et d'analyse sont prêtes !")
```
</details>
</div>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
1. Prétraitement et Qualité des données (Score de Hopkins)
</h2>

### 1.a Score de Hopkins

Le score de Hopkins génère (m) points aléatoire uniformes W et choisit (m) points réels U. 

```math
H = \frac{\sum_{i=1}^{m} w_i^d}{\sum_{i=1}^{m} u_i^d + \sum_{i=1}^{m} w_i^d}
```
Si le score H est supérieur à 0,75 on considère que les données ont une tendance au clustering. Autour des 0,5, les données sont distribuées de manière aléatoire 

```python
def hopkins_statistic(X, m_ratio=0.1):
    d = X.shape[1]
    n = len(X)
    m = int(m_ratio * n)

    rand_X = X.sample(m)
    neigh = NearestNeighbors(n_neighbors=1).fit(X)

    u_dist, _ = neigh.kneighbors(rand_X)

    min_vals, max_vals = X.min(), X.max()
    rand_uniform = np.random.uniform(low=min_vals, high=max_vals, size=(m, d))
    w_dist, _ = neigh.kneighbors(rand_uniform)

    return np.sum(w_dist) / (np.sum(u_dist) + np.sum(w_dist))
```
### 1.b Métrique d'évaluation 





| Métrique | Description | Formule | Interprétation |
| :--- | :--- | :--- | :--- |
| **Silhouette (S)** | Mesure la cohésion interne et la séparation avec les voisins. | $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$ | **Proche de 1** : Excellent<br>**Proche de 0** : Chevauchement<br>**Négatif** : Erreur d'affectation |
| **Davies-Bouldin (DB)** | Ratio de la dispersion intra-cluster sur la distance inter-clusters. | $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)$$ | **Plus bas est mieux**<br>Indique des clusters denses et bien espacés. |
| **Calinski-Harabasz (CH)** | Ratio de la variance entre les clusters et de la variance intra-cluster. | $$CH = \frac{SS_B}{SS_W} \times \frac{N - k}{k - 1}$$ | **Plus élevé est mieux**<br>Valorise la séparation nette et la compacité. |
<br>
> **Note technique** : Pour le calcul du **Score Global** final dans le benchmark, l'indice de Davies-Bouldin est inversé. Cela permet d'harmoniser les critères afin que, pour les trois métriques, une valeur élevée soit systématiquement synonyme d'un meilleur partitionnement.

```python
def evaluate_clustering(X, labels):
    if len(set(labels)) <= 1:
        return np.nan, np.nan, np.nan

    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels)
    )
```

#### Stabilité (ARI)

On ré-échantillone les données avec un bootstrapping à 80% et on compare les résultats. Ça mesure la similarité entre deux partitionnements en ajustant l'effet du hasard

```python
def compute_stability(X, model, n_runs=5):
    labels_list = []

    for i in range(n_runs):
        X_sample = resample(X, n_samples=int(0.8 * len(X)), random_state=42 + i)

        if isinstance(model, GaussianMixture):
            labels = model.fit(X_sample).predict(X_sample)
        else:
            labels = model.fit_predict(X_sample)

        labels_list.append(labels)

    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            min_len = min(len(labels_list[i]), len(labels_list[j]))
            ari = adjusted_rand_score(
                labels_list[i][:min_len],
                labels_list[j][:min_len]
            )
            ari_scores.append(ari)

    return np.mean(ari_scores)
```

### 1.c Calcul du K automatique 

On utilise un algorithme `KMeans` pour chaque valeur de k possible.

Pour chaque k, on calcule les trois métriques. Comme les résultats du score sont d'échelles différents, on fait une inversion du Davies-Bouldin. On multiplie par -1 car pour Silhouette et CH, "plus grand est mieux", alors que pour DB, "plus petit est mieux". En l'inversant, on harmonise : plus c'est grand, mieux c'est pour les trois. On ramène toutes les valeurs entre 0 et 1. Ainsi, chaque indice a le même poids dans la décision finales.

On calcule la moyenne des 3 indices normalises et on récupère la valeur de k correspondante. 

```python
def find_best_k(X, k_range):
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        sil, db, ch = evaluate_clustering(X, labels)

        results.append([k, sil, db, ch])

    df = pd.DataFrame(results, columns=["k", "sil", "db", "ch"])

    # Normalisation
    scaler = MinMaxScaler()
    scores = df[["sil", "db", "ch"]].copy()
    scores["db"] = -scores["db"]
    scores_scaled = scaler.fit_transform(scores)

    df["score"] = scores_scaled.mean(axis=1)

    best_k = df.loc[df["score"].idxmax(), "k"]

    return int(best_k), df
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
2. Application des données 
</h2>


### 2.a Chargement des données et normalisation

```python
data = pd.read_csv("[Cluster].csv")
X = data.select_dtypes(include=['float64', 'int64'])

hopkins_before = hopkins_statistic(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hopkins_after = hopkins_statistic(pd.DataFrame(X_scaled, columns=X.columns))

print(f'Hopkins avant : {hopkins_before:.3f}')
print(f'Hopkins après : {hopkins_after:.3f}')

X_best = X_scaled if hopkins_after > hopkins_before else X
```
### 2.b Réduction des données (PCA) et choix du k 

Une fois que ltest de Hopkins a validé la présence de cluster, on simplifie les données. On compresse les marqieurs pour ne garder que ceux qui expliquent 90% de la variance. Cela permet d'éliminer le bruit de fond des capteurs avant de passer aux modèles lourds

```python
#PCA
pca = PCA(n_components=0.9)
X_reduced = pca.fit_transform(X_best)

#Choix de k 
k_range = range(2, 10)
optimal_k, k_results = find_best_k(X_reduced, k_range)

print(f"\n👉 k optimal détecté : {optimal_k}")
```

### 2.c Benchmark des modèles

On lance la compétition entre les algorithmes : KMeabs, Agglomerative, Spectral, GMM, DBSCAN. 

```python
clustering_methods = {
    'KMeans': KMeans(n_clusters=optimal_k, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
    'Spectral': SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42),
    'GMM': GaussianMixture(n_components=optimal_k, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}
```
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
3. Evaluation des résultats 
</h2>


```python
results = []
for name, model in clustering_methods.items():
    try:
        if name == "GMM":
            labels = model.fit(X_reduced).predict(X_reduced)
        else:
            labels = model.fit_predict(X_reduced)
        sil, db, ch = evaluate_clustering(X_reduced, labels)
        stability = compute_stability(X_reduced, model)

    except Exception as e:
        print(f"Erreur {name}: {e}")
        sil, db, ch, stability = np.nan, np.nan, np.nan, np.nan

    results.append({
        "Méthode": name,
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch,
        "Stabilité (ARI)": stability
    })

results_df = pd.DataFrame(results)
```
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
4. Visualisation et algorithme optimal
</h2>

```python
plt.style.use('dark_background')

methods = results_df['Méthode']
x = np.arange(len(methods))
width = 0.25 

fig, ax1 = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#000000')
ax1.set_facecolor('#050505')

# Premier axe Y : Calinski-Harabasz
ch_values = results_df['Calinski-Harabasz'].fillna(0)
rects1 = ax1.bar(x - width, ch_values, width, 
                 label='Calinski-Harabasz', color='#31905e', alpha=0.8)
ax1.set_ylabel('Calinski-Harabasz (↑ mieux)', color='white', fontsize=12)
ax1.tick_params(axis='y', labelcolor='white')
ax1.set_ylim(0, ch_values.max() * 1.2 if ch_values.max() > 0 else 100)

# Deuxième axe Y : Silhouette & Davies-Bouldin
ax2 = ax1.twinx()
sil_values = results_df['Silhouette'].fillna(0)
db_values = results_df['Davies-Bouldin'].fillna(0)
rects2 = ax2.bar(x, sil_values, width, 
                 label='Silhouette', color='#9381cf', alpha=0.8)
rects3 = ax2.bar(x + width, db_values, width, 
                 label='Davies-Bouldin', color='#d67b6f', hatch='//', alpha=0.8)
ax2.set_ylabel('Silhouette (↑) & Davies-Bouldin (↓)', color='white', fontsize=12)
ax2.tick_params(axis='y', labelcolor='white')
ax2.set_ylim(0, max(sil_values.max(), db_values.max()) * 1.2 if max(sil_values.max(), db_values.max()) > 0 else 1.5)

# Légendes
plt.title("Comparaison des Scores de Clustering", color='white', fontsize=15, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, color='white')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='#222')

plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.tight_layout()
plt.show()

# Calcul de la meilleur méthode

scaler = MinMaxScaler()

scores_to_rank = results_df[['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Stabilité (ARI)']].copy()
scores_to_rank = scores_to_rank.fillna(0) 

# Attention à l'inversion de DB !

scores_to_rank['Davies-Bouldin'] = -scores_to_rank['Davies-Bouldin']

# Normalisation
scores_scaled = scaler.fit_transform(scores_to_rank)
results_df['Score_Global'] = scores_scaled.mean(axis=1)

best_method = results_df.loc[results_df['Score_Global'].idxmax(), 'Méthode']

print("\n--- Tableau récapitulatif ---")
print(results_df[['Méthode', 'Silhouette', 'Davies-Bouldin', 'Score_Global']].to_string(index=False))

print(f"\n🏆 L'algorithme recommandé pour vos données MACSima est : {best_method}")

plt.style.use('default')
```

L'absence de données pour l'algorithme DBSCAN s'explique par sa différence fondamentale de fonctionnement par rapport aux modèles basés sur les centroïdes (comme KMeans). Contrairement à ces derniers qui forcent chaque point à appartenir à un groupe, DBSCAN est un algorithme basé sur la densité. Il nécessite deux paramètres cruciaux : le rayon de recherche (eps) et le nombre minimum de points (min_samples). Si ces paramètres sont trop restrictifs par rapport à la distribution spatiale des données — notamment après une réduction de dimension (PCA) qui modifie les distances — l'algorithme peut classer la totalité des points comme du "bruit" (outliers). Mathématiquement, si aucun cluster n'est formé ou si un seul groupe global est identifié, les métriques de validation (Silhouette, Davies-Bouldin, Calinski-Harabasz) ne peuvent pas être calculées, car elles nécessitent la comparaison d'au moins deux partitions distinctes. Dans un contexte de biologie tissulaire, cela arrive souvent lorsque la densité cellulaire est trop hétérogène pour être capturée par un rayon de recherche unique et fixe.