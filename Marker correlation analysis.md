<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
Marker correlation analysis (MACSima)
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  Par <strong>Mathis BOUVET</strong> — Biologiste spécialiste en Reproduction et Développement
  <br>
  <span style="font-size: 0.8em; font-style: italic;">Mars 2026</span>
</div>

> **Note importante**
> : Ce document ne contient aucune donnée réel


<span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Issu d'une coupe tissulaire</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Code Python</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">MACSima</span> <span style="background-color: #007acc; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">Corrélation</span>

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>La corrélation</strong><br>
 La corrélation de Pearson constitue un indicateur statistique fondamental pour quantifier la relation linéaire entre l'intensité d'expression de deux marqueurs au sein d'une même unité de segmentation cellulaire. Le choix du seuil de corrélation, fixé ici à 0.30, est déterminant pour discriminer les signaux biologiques pertinents du bruit de fond instrumental ou des marquages non spécifiques. Un seuil élevé, typiquement supérieur à 0.50, est privilégié pour confirmer une co-expression stricte au sein d'un même compartiment cellulaire ou pour valider l'identité d'un phénotype précis. À l'inverse, un seuil plus modéré compris entre 0.20 et 0.50 permet de capturer des interactions plus subtiles, comme des protéines fonctionnelles partageant un micro-environnement commun ou des marqueurs de voisinage immédiat. Il est toutefois crucial de considérer que ces coefficients peuvent être influencés par la précision de la segmentation initiale : une délimitation cellulaire trop permissive peut artificiellement corréler des signaux provenant de cellules adjacentes, tandis qu'une hétérogénéité tissulaire marquée peut diluer une corrélation forte localisée dans une structure spécifique du tissu.
</div>
<br>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objectif
</h2>
L'objectif est d'identifier statistiquement quels marqueurs partagent un profil d'expression similaire à l'échelle cellulaire, ce qui permet de définir des signatures phénotypiques ou des voisinages cellulaires.

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
    "matplotlib": "matplotlib",
    "pandas": "pandas",
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
import pandas as pd 
import matplotlib.pyplot as plt

print("✨ Tous les modules sont importés et prêts à l'emploi !")
```
</details>
</div>

## 1. Chargement des données

```python
df = pd.read_csv(r"[Fichier_analyse].csv")
pd.set_option("display.max_columns", None)
print(df.head())
```
## 2. Calcul de la corrélation

Bien que la corrélation de Pearson soit la mesure de référence pour les relations linéaires, d'autres approches statistiques comme les corrélations de Spearman ou de Kendall offrent une robustesse accrue face à l'hétérogénéité des données d'imagerie. La corrélation de Spearman, en se basant sur les rangs plutôt que sur les valeurs absolues d'intensité, permet de capturer des relations monotones non linéaires tout en minimisant l'impact des valeurs aberrantes ou des artefacts de saturation fréquents en fluorescence. De même, le coefficient de Kendall propose une analyse de concordance rigoureuse, particulièrement adaptée lorsque la distribution des marqueurs est fortement asymétrique. Le choix entre ces méthodes dépend donc de la distribution statistique des signaux : une relation strictement proportionnelle sera mieux décrite par Pearson, tandis qu'une tendance biologique globale, potentiellement affectée par un bruit de fond non uniforme, sera plus fidèlement représentée par une approche non paramétrique comme celle de Spearman.

Ici on utilise une comparaison entre 3 coefficients de corrélation utilisé en biologie `Pearson`, `Spearman`, `Kendal`.

Ce qui est intéressant lorsqu'on compare beaucoup de marqueur, par exemple lorsqu'on a screnner des marqueurs immuntaire c'est que les comparaisons entre les corrélation peuvent apporter quelques détails. Si un marqueur chute au classement avec Spearman, c'est probablement qu'il y a avait quelques cellules avec une expression extrêmement forte qui tirait artificiellement la corrélation de Pearson vers le haut 

```python
plt.style.use('dark_background')

target = "(marqueur à analyser) Cell Exp"
methods = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}
results = {m: [] for m in methods}

def get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""
for col in df.columns:
    if col != target and df[col].dtype in ['float64', 'int64']:
        mask = df[col].notna() & df[target].notna()
        if mask.sum() > 2:
            for m_name, m_func in methods.items():
                corr, p_val = m_func(df[col][mask], df[target][mask])
                if corr > 0.30: # Ton seuil de force
                    results[m_name].append({
                        'Marker': col, 
                        'Corr': corr, 
                        'p': p_val, 
                        'Stars': get_stars(p_val)
                    })

fig, axes = plt.subplots(1, 3, figsize=(22, 12), sharey=False)
fig.suptitle(f"Analyse Multi-Statistique (Réf: {target})", fontsize=20, weight="bold", y=0.98)

for i, (m_name, data_list) in enumerate(results.items()):
    if data_list:
        res_df = pd.DataFrame(data_list).sort_values('Corr', ascending=True)
        labels = [f"{row['Marker']} {row['Stars']}" for _, row in res_df.iterrows()]
        color_map = plt.cm.viridis(res_df['Corr'])
        axes[i].barh(labels, res_df['Corr'], color=color_map, edgecolor="white", linewidth=0.6)
        axes[i].set_title(f"Méthode : {m_name.upper()}", fontsize=15, pad=15, color="cyan")
        axes[i].axvline(0.3, color="red", linestyle="--", alpha=0.6, label="Seuil r=0.30")
        axes[i].set_xlabel("Coefficient de corrélation")
        axes[i].grid(axis='x', linestyle=':', alpha=0.2)
    else:
        axes[i].text(0.5, 0.5, "Aucune corrélation > 0.30", ha='center', va='center', fontsize=14)
legend_text = (
    "Significativité (p-value) :\n"
    "*** : p < 0.001 (Extrêmement significatif)\n"
    "** : p < 0.01  (Très significatif)\n"
    "* : p < 0.05  (Significatif)\n"
    "Seuil de corrélation minimum : 0.30"
)

fig.text(0.85, 0.02, legend_text, fontsize=11, color="white",
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=1', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.show()
```