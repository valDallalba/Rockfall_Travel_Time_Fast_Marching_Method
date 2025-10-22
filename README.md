# 🪨 Simulation de propagation de blocs rocheux par Fast Marching Method (FFM)

## 📘 Description du projet

Ce projet vise à estimer les **temps de parcours potentiels des impacts rocheux** depuis des **zones sources** sur un **modèle numérique de terrain (MNT / DEM)**, en utilisant une implémentation efficace de la **méthode de Fast Marching (FFM)**.  
Cette approche permet de modéliser la propagation d’une onde de "temps de parcours" sur le terrain, en tenant compte de la topographie et d’un champ de vitesses dérivé des pentes locales.

L’objectif final est de produire :

- 🗺️ Une **carte raster du temps de parcours** (en secondes) depuis les zones sources,  
- 🧩 Une **classification des points d’impact** selon leur temps d’arrivée (déciles ou pourcentages),  
- 📈 Des **cartes d’isochrones** et **zones d’impact probables**, exportées sous forme de rasters ou shapefiles.

---

## ⚙️ Fonctionnement général

L’algorithme suit les étapes suivantes :

1. **Chargement et préparation des données**  
   - Découpage du MNT à la zone d’étude.  
   - Chargement des zones sources et des points d’impact (GeoDataFrame).  
   - Harmonisation des systèmes de coordonnées (CRS).

2. **Création du champ de vitesse**  
   - Calcul de la pente depuis le DEM (`dem_to_slope`).  
   - Conversion de la pente en vitesse de déplacement à l’aide d’une fonction empirique (`slope_to_fall_speed`).

3. **Application de la Fast Marching Method**  
   - Propagation du front de temps depuis les zones sources (`fast_marching_multi`).  
   - Résolution locale de l’équation d’Eikonal (`solve_eikonal`) pour chaque cellule.

4. **Extraction et classification des temps de parcours**  
   - Extraction du temps de parcours pour chaque point d’impact (`extract_travel_time_for_points`).  
   - Classement par déciles et attribution d’un rang (`classify_by_deciles`, `rank_by_travel_time`).

5. **Génération des cartes d’isochrones**  
   - Transformation du raster de temps en classes de pourcentage (5%, 10%, …, 100%) (`transform_to_isochrone_map`).  
   - Export en shapefile via `isochrone_to_shp`.

6. **Analyse des zones d’impact**  
   - Comptage du nombre d’impacts par cellule de grille à une résolution donnée (`impacts_count`).  
   - Calcul de la probabilité d’impact par maille.

---

## 🧠 Contenu du dépôt

| Fichier | Rôle |
|----------|------|
| `functions_Rock_fall_FFM.py` | Ensemble des fonctions principales : chargement des données, génération du champ de vitesses, algorithme de Fast Marching, classification et export des résultats. |
| `rock_fall_classification_method.ipynb` | Notebook de démonstration illustrant le workflow complet sur un jeu de données exemple : préparation des entrées, exécution du modèle, visualisation des résultats. |

---

## 📤 Sorties produites

Le pipeline produit les outputs suivants :

- 🕒 **Raster de temps de parcours** : temps minimal (en secondes) pour atteindre chaque cellule depuis les zones sources.  
- 🗺️ **Carte d’isochrones** : classes de 5% du temps de parcours (0–5%, 5–10%, …, 95–100%).  
- 📁 **Shapefile d’isochrones** : polygones correspondant aux intervalles de pourcentages.  
- 📊 **GeoDataFrame des points d’impact classés** : attributs `travel_time`, `decile`, `rank`.  
- 📦 **Grille d’impacts agrégés** : shapefile avec champs `count` (nombre d’impacts) et `probability` (probabilité d’impact).

---

## 🧩 Dépendances principales

Le projet repose sur les bibliothèques suivantes :

```bash
numpy
geopandas
rasterio
shapely
matplotlib
