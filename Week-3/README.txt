# Iris Flower Clustering Project — Submission Folder

## Files to submit
1. `iris_clustering_project.ipynb` — Jupyter/Colab notebook
2. `iris_clustering_project.py` — standalone Python script
3. `Iris.csv` — dataset used by the project
4. `Iris_KMeans_Results.csv` — generated cluster assignments
5. `01_kmeans_clusters.png` — K-Means visualization
6. `02_pca_2d.png` — PCA 2D visualization
7. `VERIFICATION_RESULTS.txt` — cross-checked numerical results

## Requirements covered
- K-Means clustering with k=3
- Cluster visualization
- PCA reduction to 2D
- Explained variance ratio
- Comparison of predicted clusters with true labels

## Reproducibility
K-Means uses `random_state=42` and `n_init=10`.
The four measurement columns are used as features.
`Id` is excluded because it is an identifier, and `Species` is excluded from
clustering because it is the true label used only for comparison.


## Project conclusion
The project includes a written conclusion in the notebook and a conclusion summary in the Python script.
The results show meaningful agreement between K-Means clusters and the true Iris species labels, while PCA reduces the four features to two components explaining approximately 95.80% of the variance.

Note: K-Means cluster labels are arbitrary numerical identifiers and do not directly represent Iris species.
