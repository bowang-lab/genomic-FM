import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import time


def plot_clusters(vectors, labels, assignments, filename="test_cluster_visualization.pdf"):
    """
    Visualizes the correlation between cluster assignments and true labels using a confusion matrix.

    Parameters:
    - vectors: array-like, shape (n_samples, n_features)
      The data points to be plotted.
    - labels: list or array, shape (n_samples,)
      The ground truth labels for the data points.
    - assignments: list or array, shape (n_samples,)
      The cluster assignments for the data points.
    - use_tsne: bool, optional (default=False)
      Whether to use t-SNE to reduce dimensionality to two dimensions.
    - filename: str, optional (default="test_cluster_visualization.pdf")
      The filename where the plot will be saved.
    """

    # Convert labels and assignments to string to avoid data type issues
    labels = np.array(labels).astype(str)
    assignments = np.array(assignments).astype(str)

    # Compute confusion matrix
    cm = confusion_matrix(labels, assignments)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=np.unique(assignments), yticklabels=np.unique(labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Cluster Assignments')
    plt.ylabel('True Labels')

    plt.savefig(filename)
    # plt.show()

def plot_clusters_scatter(vectors, labels, filename="plot_clusters_scatter.pdf"):
    """
    Visualizes the 2D distribution of the data using PCA and labels the points according to the real labels.

    Parameters:
    - vectors: array-like, shape (n_samples, n_features)
      The data points to be plotted.
    - labels: list or array, shape (n_samples,)
      The ground truth labels for the data points.
    - filename: str, optional
      The filename where the plot will be saved.
    """
    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)

    # Generate a continuous colormap and then create a discrete colormap from it
    continuous_cmap = plt.cm.get_cmap('tab20', 256)  # Switched to 'tab20' for more distinct colors
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_unique_labels)))  # Corrected linspace range

    # Map each unique label to a specific color
    label_to_color = dict(zip(unique_labels, discrete_cmap.colors))
    color_labels = [label_to_color[label] for label in labels]
    labels = np.array(labels)
    # Perform PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(6, 6))
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)
        plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], label=label, color=discrete_cmap.colors[i], alpha=0.6)

    plt.title('2D Distribution of Data with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)

def compute_confusion_matrix(labels_int, labels_str):
    # Convert integer labels and string labels to numpy arrays if they aren't already
    labels_int = np.array(labels_int)
    labels_str = np.array(labels_str)

    # Ensure that both arrays have the same length
    if len(labels_int) != len(labels_str):
        raise ValueError("Both input arrays must have the same number of elements.")

    # Create mappings for integer and string labels to a common categorical range
    unique_int_labels, labels_int_mapped = np.unique(labels_int, return_inverse=True)
    unique_str_labels, labels_str_mapped = np.unique(labels_str, return_inverse=True)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_int_mapped, labels_str_mapped)

    return conf_matrix, unique_int_labels, unique_str_labels


def compute_and_save_confusion_matrix(labels_int, labels_str, filename='confusion_matrix.pdf'):
    # Convert integer labels and string labels to numpy arrays if they aren't already
    labels_int = np.array(labels_int)
    labels_str = np.array(labels_str)

    # Ensure that both arrays have the same length
    if len(labels_int) != len(labels_str):
        raise ValueError("Both input arrays must have the same number of elements.")

    # Create mappings for integer and string labels to a common categorical range
    unique_int_labels, labels_int_mapped = np.unique(labels_int, return_inverse=True)
    unique_str_labels, labels_str_mapped = np.unique(labels_str, return_inverse=True)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_int_mapped, labels_str_mapped)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_str_labels, yticklabels=unique_int_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the plot to a PDF file
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return conf_matrix, unique_int_labels, unique_str_labels




from src.variants_vector_index.vector_loader import VectorLoader

vec_loader = VectorLoader(dataset='real_clinvar_hyena-tiny')
# vec_loader = VectorLoader(dataset='real_clinvar_dnabert2',
#                           checkpoint='/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/eyrvqq7f/checkpoints/epoch=99-step=431100.ckpt')
# vec_loader = VectorLoader(dataset='real_clinvar_dnabert2')
######
# test cluster visualization
######
print(f"num of classes in labels: {len(set(vec_loader.labels))}")
plot_clusters_scatter(vec_loader.vectors, vec_loader.labels)

assignments, dis = vec_loader.perform_clustering(n_clusters=5)
compute_and_save_confusion_matrix(vec_loader.labels, assignments)
# plot_clusters(vec_loader.vectors, vec_loader.labels, assignments)


#######
# Test query_vectors
######

query_vector = vec_loader.vectors[1]
query_vector_label = vec_loader.labels[1]
start_time = time.time()
distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=20)
end_time = time.time()
query_time = end_time - start_time
# mapped_result_labels = np.array([i.item() for i in result_labels])
mapped_result_labels = result_labels
print(f"Unique Counts: {np.unique(mapped_result_labels, return_counts=True)}")
print(f"Query vector label: {query_vector_label}")
print(f"Distances: {distances}")
print(f"Result labels: {result_labels}")
print("========================")
print(f"Query vector size: {query_vector.shape}")
print(f"Query time: {query_time} seconds")
