import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np

def plot_clusters(vectors, labels, assignments, use_tsne=False, filename="test_cluster_visualization.pdf"):
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

    if use_tsne:
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        vectors = tsne.fit_transform(vectors)
        print("t-SNE application complete.")

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
    plt.show()


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

vec_loader = VectorLoader(from_cache=True)

######
# test cluster visualization
######
print(f"num of classes in labels: {len(set(vec_loader.labels))}")

assignments, dis = vec_loader.perform_clustering(n_clusters=4)
compute_and_save_confusion_matrix(vec_loader.labels, assignments)
# plot_clusters(vec_loader.vectors, vec_loader.labels, assignments)


#######
# Test query_vectors
######

query_vector = vec_loader.vectors[1]
query_vector_label = vec_loader.labels[1]
distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=1)
mapped_result_labels = np.array([i.item() for i in result_labels])
print(f"Unique Counts: {np.unique(mapped_result_labels, return_counts=True)}")
print(f"Query vector label: {query_vector_label}")
print(f"Distances: {distances}")
print(f"Result labels: {result_labels}")
