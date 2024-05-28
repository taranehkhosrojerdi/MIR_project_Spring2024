# import numpy as np
# import os
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# import sys
# sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

# from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
# from Logic.core.word_embedding.fasttext_model import FastText
# from Logic.core.clustering.dimension_reduction import DimensionReduction
# from Logic.core.clustering.clustering_metrics import ClusteringMetrics
# from Logic.core.clustering.clustering_utils import ClusteringUtils

# import matplotlib.pyplot as plt
# import wandb
# wandb.login(key='39f85cb4fa0204703a18a1c31cced6f4fc2be8fe')

# # Main Function: Clustering Tasks

# # 0. Embedding Extraction
# # TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
# file_path = 'Logic/tests/dumb_IMDB_crawled.json'
# loader = FastTextDataLoader(file_path)
# loader.read_data_to_df()
# out = loader.create_train_data()
# X, y = out[0], out[1]

# ft_model = FastText(method='skipgram')
# ft_model.train(X.tolist())

# def get_cleaned_embeddings(data, model):
#     data = data.astype(str)
#     cleaned_data = np.char.replace(data, '\n', '')
#     embeddings = np.array([model.get_query_embedding(text) for text in cleaned_data])
#     return np.array(embeddings)

# X_embeddings = get_cleaned_embeddings(X, ft_model)
# y_embeddings = get_cleaned_embeddings(y, ft_model)

# # 1. Dimension Reduction
# # TODO: Perform Principal Component Analysis (PCA):
# #     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
# #     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
# #     - Draw plots to visualize the results.
# dimension_reduction = DimensionReduction()
# reduced_embeddings = dimension_reduction.pca_reduce_dimension(embeddings=X_embeddings, n_components=100) # TODO
# dimension_reduction.wandb_plot_explained_variance_by_components(data=X_embeddings, project_name='dumb_PCA', run_name='dumb_PCA')

# # TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
# #     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
# #     - Use the output vectors from this step to draw the diagram.
# tsne_2d_embeddings = dimension_reduction.convert_to_2d_tsne(emb_vecs=reduced_embeddings)
# dimension_reduction.wandb_plot_2d_tsne(data=reduced_embeddings, project_name='dumb_2TSNE', run_name='dumb_2TSNE')

# # 2. Clustering
# ## K-Means Clustering
# # TODO: Implement the K-means clustering algorithm from scratch.
# # TODO: Create document clusters using K-Means.
# # TODO: Run the algorithm with several different values of k.
# # TODO: For each run:
# #     - Determine the genre of each cluster based on the number of documents in each cluster.
# #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
# #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# # TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# # TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

# def initialize_centroids(data, k):
#     indices = np.random.choice(data.shape[0], k, replace=False)
#     return data[indices]

# def assign_clusters(data, centroids):
#     distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
#     return np.argmin(distances, axis=1)

# def update_centroids(data, labels, k):
#     centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
#     return centroids

# def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
#     centroids = initialize_centroids(data, k)
#     for i in range(max_iters):
#         labels = assign_clusters(data, centroids)
#         new_centroids = update_centroids(data, labels, k)
#         if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
#             break
#         centroids = new_centroids
#     return labels, centroids


# def plot_clusters(data, labels, title):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
#     plt.title(title)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.show()

# ks = [10, 100, 500]
# silhouette_scores = []
# purity_values = []
# reduced_embeddings = np.array(reduced_embeddings)

# clustering_metrics = ClusteringMetrics()

# for k in ks:
#     labels, centroids = kmeans_clustering(reduced_embeddings, k)
#     plot_clusters(np.array(tsne_2d_embeddings), labels, f"K-Means Clustering with k={k}")
#     silhouette_avg = clustering_metrics.silhouette_score(reduced_embeddings, labels)
#     silhouette_scores.append(silhouette_avg)
#     purity = clustering_metrics.purity_score(y, labels)
#     purity_values.append(purity)
#     wandb.log({f"Silhouette Score (k={k})": silhouette_avg, f"Purity (k={k})": purity})

# plt.figure(figsize=(10, 7))
# plt.plot(ks, silhouette_scores, marker='o', linestyle='--')
# plt.title("Silhouette Score for Different Values of k")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.plot(ks, purity_values, marker='o', linestyle='--')
# plt.title("Purity for Different Values of k")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Purity")
# plt.show()

# wandb.log({"Silhouette Scores": wandb.Image(plt.figure(1)), "Purity Values": wandb.Image(plt.figure(2))})


# ## Hierarchical Clustering
# # TODO: Perform hierarchical clustering with all different linkage methods.
# # TODO: Visualize the results.
# data = np.array(reduced_embeddings)

# utils = ClusteringUtils()

# linkage_methods = ['single', 'complete', 'average', 'ward']
# for method in linkage_methods:
#     utils.wandb_plot_hierarchical_clustering_dendrogram(data, project_name='your_project', linkage_method=method, run_name=f'hierarchical_{method}')

# # 3. Evaluation
# # TODO: Using clustering metrics, evaluate how well your clustering method is performing.

# best_k = ks[np.argmax(silhouette_scores)]
# best_labels, _ = kmeans_clustering(reduced_embeddings, best_k)

# silhouette_avg = clustering_metrics.silhouette_score(reduced_embeddings, best_labels)
# purity = clustering_metrics.purity_score(y, best_labels)
# adjusted_rand = clustering_metrics.adjusted_rand_score(y, best_labels)

# print(f"Best k (based on silhouette score): {best_k}")
# print(f"Silhouette Score: {silhouette_avg}")
# print(f"Purity: {purity}")
# print(f"Adjusted Rand Index: {adjusted_rand}")

# wandb.log({"Final Silhouette Score": silhouette_avg, "Final Purity": purity, "Final Adjusted Rand Index": adjusted_rand})
