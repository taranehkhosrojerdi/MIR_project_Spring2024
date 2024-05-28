import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
from .clustering_metrics import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        kmeans.fit(emb_vecs)
        return kmeans.cluster_centers_, kmeans.labels_

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc.split())
        return word_counts.most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """
        This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
        kmeans.fit(emb_vecs)
        wcss = kmeans.inertia_
        return kmeans.cluster_centers_, kmeans.labels_, wcss

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        hc = AgglomerativeClustering(linkage='single')
        return hc.fit_predict(emb_vecs)

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        hc = AgglomerativeClustering(linkage='complete')
        return hc.fit_predict(emb_vecs)

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        hc = AgglomerativeClustering(linkage='average')
        return hc.fit_predict(emb_vecs)

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        hc = AgglomerativeClustering(linkage='ward')
        return hc.fit_predict(emb_vecs)

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """
        This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        labels = kmeans.labels_

        # Convert to 2D using TSNE
        tsne = TSNE(n_components=2)
        data_2d = tsne.fit_transform(data)

        plt.figure(figsize=(10, 8))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title(f'K-Means Clustering with {n_clusters} Clusters')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')

        wandb.log({"KMeans Clustering": plt})

        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """
        This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)

        # Perform hierarchical clustering
        linked = linkage(data, method=linkage_method)

        # Create linkage matrix for dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')

        # Log the plot to wandb
        wandb.log({"Hierarchical Clustering Dendrogram": plt})

        plt.close()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None, run_name=None):
        """
        This function calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(embeddings, labels))
            purity_scores.append(purity_score(true_labels, labels))

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, label='Silhouette Score', marker='o')
        plt.plot(k_values, purity_scores, label='Purity Score', marker='x')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Score')
        plt.title('Clustering Scores for Different k Values')
        plt.legend()

        if project_name and run_name:
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Clustering Scores": plt})

        plt.show()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None, run_name=None):
        """
        This function calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(embeddings, labels))
            purity_scores.append(purity_score(true_labels, labels))

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, label='Silhouette Score', marker='o')
        plt.plot(k_values, purity_scores, label='Purity Score', marker='x')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Score')
        plt.title('Clustering Scores for Different k Values')
        plt.legend()

        if project_name and run_name:
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Clustering Scores": plt})

        plt.show()
        