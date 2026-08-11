# 🏦 Credit Card Customer Segmentation using Unsupervised Machine Learning

Segmenting ~9,000 credit card customers into behavioral groups using **K-Means** and **DBSCAN**, so a business can design targeted marketing, retention, and credit-risk strategies instead of treating every customer the same.

## Problem Statement
A credit card company treats its entire customer base as one group, which is inefficient — a customer who pays in full and rarely uses the card has very different needs from one who revolves a balance and relies on cash advances. This project uses unsupervised clustering to find natural customer segments in 18 months of usage data (balance, purchases, cash advances, credit limit, payment behavior, tenure).

## Dataset
[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) — `CC_GENERAL.csv`, ~9,000 rows, 18 behavioral features per customer.

## Approach
1. Data cleaning (duplicates, missing values via median imputation)
2. Feature scaling with `StandardScaler`
3. Optimal K selection via Elbow Method + Silhouette Score
4. K-Means clustering + PCA visualization
5. Cluster profiling (relative feature importance per segment)
6. DBSCAN as a density-based alternative, with `eps` hyperparameter tuning
7. Model comparison and business-insight generation per cluster

## Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` (KMeans, DBSCAN, PCA, StandardScaler, silhouette_score)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook CC_General_Customer_Segmentation.ipynb
```
Or open directly in [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code).

## Results
- K-Means (K=4) produced well-separated, evenly-sized, actionable segments.
- DBSCAN independently isolates outlier/noise customers that don't fit any clear pattern.
- Full model comparison and per-cluster business interpretation are in the notebook.

## License
MIT
