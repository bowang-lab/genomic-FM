import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import time


from src.variants_vector_index.vector_loader import VectorLoader


DATASETS = ['clinvar_CLNSIG_hyena-tiny']
# DATASETS = ['real_clinvar_dnabert2', 'real_clinvar_hyena-tiny']
# Finetunes = [False]
# DATASETS = ['real_clinvar_nt','real_clinvar_ntv2']
Finetunes = [True]

# DATASETS = ['real_clinvar_random']
# Finetunes = [True]
CHECKPOINT_DIR = {
    'real_clinvar_hyena-tiny': '/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/luxnk59q/checkpoints/epoch=99-step=431100.ckpt',
    'clinvar_CLNSIG_hyena-tiny': '/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/luxnk59q/checkpoints/epoch=99-step=431100.ckpt',
    'real_clinvar_dnabert2': '/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/eyrvqq7f/checkpoints/epoch=99-step=431100.ckpt',
    'real_clinvar_nt':'/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/8ae16qsr/checkpoints/epoch=99-step=431100.ckpt',
    'real_clinvar_ntv2': '/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Run-GFM/s3c0cdfc/checkpoints/epoch=99-step=431100.ckpt',
}

#######
# Test query_vectors
######

# def compute_query_accuracy(vectors_list, labels_list, vec_loader):
#     correct = 0
#     total = 0
#     for i in range(len(vectors_list)):
#         query_vector = vectors_list[i]
#         query_vector_label = labels_list[i]
#         distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=10)
#         # Print the results
#         print(f"Query vector label: {query_vector_label}")
#         print(f"Distances: {distances}")
#         print(f"Result labels: {result_labels}")
#         # end of print
#         for i in range(len(result_labels)):
#           if query_vector_label == result_labels[i]:
#               correct += 1
#           total += 1
#     return correct / total
# # randomly sample n query vectors
# vectors_list = []
# labels_list = []
# n = 10
# idx = np.random.choice(len(vec_loader.vectors), n, replace=False)
# for i in idx:
#     vectors_list.append(vec_loader.vectors[i])
#     labels_list.append(vec_loader.labels[i])

# accuracy = compute_query_accuracy(vectors_list, labels_list, vec_loader)
# print(f"Accuracy of {DATASET}_{Finetune}: {accuracy}")

def compute_query_accuracy(vectors_list, labels_list, vec_loader):
    correct = 0
    total = 0
    for i in range(len(vectors_list)):
        query_vector = vectors_list[i]
        query_vector_label = labels_list[i]
        distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=10)
        for j in range(len(result_labels)):
            if query_vector_label == result_labels[j]:
                correct += 1
            total += 1
    return correct / total

def compute_query_time(query_vector_list, query_vector_label_list, vec_loader):
    correct = 0
    total = 0
    for query_vector, query_vector_label in zip(query_vector_list,query_vector_label_list):
        start = time.time()
        distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=10)
        end = time.time()
    return end - start

def sample_vectors(vec_loader, n):
    vectors_list = []
    labels_list = []
    idx = np.random.choice(len(vec_loader.vectors), n, replace=False)
    for i in idx:
        vectors_list.append(vec_loader.vectors[i])
        labels_list.append(vec_loader.labels[i])
    return vectors_list, labels_list

def compute_mean_std_accuracy(vec_loader, n=1, runs=10):
    accuracies = []
    for _ in range(runs):
        vectors_list, labels_list = sample_vectors(vec_loader, n)
        print(f"shape of query vector: {vectors_list[0].shape}")
        accuracy = compute_query_time(vectors_list, labels_list, vec_loader)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return mean_accuracy, std_accuracy

def eval(datasets,finetunes):
    for dataset in datasets:
        for finetune in finetunes:
          vec_loader = VectorLoader(dataset=dataset, checkpoint=CHECKPOINT_DIR[dataset] if finetune else None, from_cache=True)
          mean_accuracy, std_accuracy = compute_mean_std_accuracy(vec_loader, n=10, runs=5)
          print(f"Mean Query Time {dataset} ({finetune}): {mean_accuracy}")
          print(f"Standard Deviation of Query Time {dataset} ({finetune}): {std_accuracy}")
# Start of Evaluation
# vec_loader = VectorLoader(dataset=DATASET, checkpoint=CHECKPOINT_DIR[DATASET] if Finetune else None)
# mean_accuracy, std_accuracy = compute_mean_std_accuracy(vec_loader, n=10, runs=5)
# print(f"Mean Accuracy {DATASET} ({Finetune}): {mean_accuracy}")
# print(f"Standard Deviation of Accuracy {DATASET} ({Finetune}: {std_accuracy}")
eval(DATASETS, Finetunes)
