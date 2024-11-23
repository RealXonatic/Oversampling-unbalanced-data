# Rapport de Stage 2024

## Machine Learning-Based Decision Support in Ophthalmology

Ce dépôt contient le code et les résultats du stage réalisé à l'**Université Polytechnique de Catalogne (UPC)**, au sein du groupe de recherche **SOCO (Soft Computing)**, dans le cadre du projet intitulé **Eye-AI**. Ce projet vise à développer des outils d'aide à la décision basés sur l'apprentissage automatique pour analyser les images multi-modales de la vascularité rétinienne.

---

## 📋 Objectifs

- **Problématique :** Résoudre le problème de déséquilibre des classes dans des ensembles de données d'imagerie médicale afin d'améliorer les performances des modèles de machine learning.
- **Méthodes utilisées :**
  - Techniques de suréchantillonnage (SMOTE, Borderline-SMOTE, ADASYN, etc.).
  - Évaluation des performances des modèles via des métriques adaptées (Macro Accuracy).
  - Visualisation des régions de décision pour une meilleure interprétabilité.

---

## 📂 Structure du Projet

### 1. Analyse des Données
- **Description des données :** Images OCT et OCTA représentant différentes pathologies liées à la rétinopathie diabétique.
- **Problèmes abordés :**
  - **Problème B (DR)** : Diabétiques sans rétinopathie vs. diabétiques avec rétinopathie.
  - **Problème C (RFDR)** : Diabétiques avec rétinopathie légère vs. rétinopathie sévère.
- **Étapes de préparation :**
  - Prétraitement réalisé (par le tuteur).
  - Analyse des classes pour identifier les déséquilibres.

### 2. Méthodes de Suréchantillonnage
- **Techniques mises en œuvre :**
  - SMOTE (Synthetic Minority Over-sampling Technique).
  - Borderline-SMOTE.
  - ADASYN (Adaptive Synthetic Sampling).
  - Combinaisons SMOTEENN.
- **Impact :** Équilibrage des données pour réduire les biais des modèles.

### 3. Implémentation en Python
- **Bibliothèques utilisées :**
  - `imbalanced-learn` pour les méthodes de suréchantillonnage.
  - `scikit-learn` pour l'entraînement et l'évaluation des modèles.
  - `matplotlib` et `seaborn` pour les visualisations.
- **Modèles testés :**
  - Régression logistique.
  - Support Vector Machines (SVM).
  - Random Forest.

### 4. Résultats
- Visualisation des régions de décision avant et après suréchantillonnage.
- Comparaison des performances entre les modèles et les méthodes :
  - Macro Accuracy : amélioration significative après application d'ADASYN.
  - Meilleures performances obtenues avec le modèle SVM et ADASYN.

---

## 🌟 Points Forts
- Approche innovante pour traiter les déséquilibres dans des données critiques.
- Visualisation intuitive des régions de décision et des impacts des techniques.
- Contributions à un projet de recherche en ophtalmologie avec des implications réelles.

---

## 👤 Auteur
- **Jaheer Goulam**
- Tuteurs : Prof. Angela Nebot, Dr. Enrique Romero
- [Profil GitHub](https://github.com/RealXonatic)

Pour toute question ou collaboration, n'hésitez pas à me contacter !

---

## 📊 Ressources
Les données et les scripts sont disponibles dans le dépôt GitHub principal : [Oversampling-for-Ophtamologic-data](https://github.com/RealXonatic/Oversampling-for-Ophtamologic-data).
