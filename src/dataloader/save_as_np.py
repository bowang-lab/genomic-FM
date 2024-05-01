import numpy as np
import torch
import math
import os
import yaml
from tqdm import tqdm
from sklearn.decomposition import PCA


def apply_pca_torch(data, n_components=16):
    # Ensure data is a torch tensor and optionally move data to GPU
    data_tensor = torch.tensor(data).float()
    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()

    # Reshape the data
    reshaped_data = data_tensor.reshape(-1, data_tensor.shape[-1])  # Reshape to (num_samples * 1024, 128)

    # Center the data by subtracting the mean
    mean = torch.mean(reshaped_data, dim=0)
    data_centered = reshaped_data - mean

    # Compute SVD
    U, S, V = torch.svd(data_centered)

    # Compute PCA by projecting the data onto the principal components
    pca_transformed_data = torch.mm(data_centered, V[:, :n_components])

    # Reshape back to the original shape
    pca_transformed_data = pca_transformed_data.reshape(data_tensor.shape[:-1] + (-1,))  # Reshape back to (num_samples, 1024, n_components)

    # Optionally move data back to CPU
    pca_transformed_data = pca_transformed_data.cpu()

    return pca_transformed_data.numpy()  # Convert to numpy if needed


def apply_pca(data, n_components=16):
    # Reshape the data
    reshaped_data = data.reshape(-1, data.shape[-1])  # Reshape to (num_samples * 1024, 128)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(reshaped_data)
    pca_transformed_data = pca.transform(reshaped_data)

    # Reshape back to the original shape
    pca_transformed_data = pca_transformed_data.reshape(data.shape[:-1] + (-1,))  # Reshape back to (num_samples, 1024, n_components)

    return pca_transformed_data



def get_mapped_class(data, task='classification'):
    x_class = {}
    y_class = {}
    x_count = 0
    y_count = 0
    for i in tqdm(range(len(data)), desc="Mapping Annotation classes"):
        x, y = data[i]
        annotation = x[2]
        if annotation not in x_class:
            x_class[annotation] = x_count
            x_count += 1
        if task == 'classification':
            if y not in y_class:
                y_class[y] = y_count
                y_count += 1
    return x_class, y_class

def map_to_given_class(data, x_class, y_class, task='classification'):
    for i in range(len(data)):
        element = data[i]
        x, y = element
        annotation = x[2]
        x[2] = x_class[annotation]
        if task == 'classification':
            element[1] = y_class[y]
        elif task == 'regression':
            element[1] = torch.tensor([y], dtype=np.float32)
    return data

def map_to_class(data, task='classification', dataset_name="test", path='root/data/npy_output'):
    # x = (reference, alternate, annotation)
    # map x annotation to corresponding class label
    # y = target， e.g. beneign or pathogenic, slope, p_val, splice_change
    # for classification task, map y to corresponding class label
    # for regression task, keep y as it is
    # save the mapping as a file
    x_class = {}
    y_class = {}
    x_count = 0
    y_count = 0
    for i in range(len(data)):
        element = data[i]
        x, y = element
        annotation = x[2]
        if annotation not in x_class:
            x_class[annotation] = x_count
            x_count += 1
        x[2] = x_class[annotation]
        if task == 'classification':
            if y not in y_class:
                y_class[y] = y_count
                y_count += 1
            element[1] = y_class[y]
        if task == 'regression':
            # ensure y is torch tensor float
            element[1] = torch.tensor([y], dtype=np.float32)
    with open(f'{path}/{dataset_name}_x_class.yaml', 'w') as f:
        yaml.dump(x_class, f)
    with open(f'{path}/{dataset_name}_y_class.yaml', 'w') as f:
        yaml.dump(y_class, f)
    return x_class, y_class

def has_cache(cache_dir, base_filename):
    return os.path.exists(f'{cache_dir}/{base_filename}_seq1_0.npy')


def get_cache(base_filename, cache_dir='root/data/npy_output'):
    if has_cache(cache_dir, base_filename):
        print(f"Cache file {base_filename} already exists.")
        # get a list of files under the directory
        files = os.listdir(cache_dir)
        seq1_paths = [f'{cache_dir}/{f}' for f in files if f.startswith(f'{base_filename}_seq1')]
        seq2_paths = [f'{cache_dir}/{f}' for f in files if f.startswith(f'{base_filename}_seq2')]
        annot_paths = [f'{cache_dir}/{f}' for f in files if f.startswith(f'{base_filename}_annot')]
        label_paths = [f'{cache_dir}/{f}' for f in files if f.startswith(f'{base_filename}_y')]
    else:
        seq1_paths = None
        seq2_paths = None
        annot_paths = None
        label_paths = None
    # sort the paths
    seq1_paths = sorted(seq1_paths)
    seq2_paths = sorted(seq2_paths)
    annot_paths = sorted(annot_paths)
    label_paths = sorted(label_paths)
    return seq1_paths, seq2_paths, annot_paths, label_paths

def save_data(data, base_filename='data', base_index=0, base_dir='root/data/npy_output',pca_components=16):

    base_filename = f'{base_dir}/{base_filename}'
    seq1_paths = None
    seq2_paths = None
    annot_paths = None
    label_paths = None
    # Check if the data list is empty
    if not data:
        raise ValueError("Data list is empty.")

    num_items = len(data)
    print(f"Saving {num_items} items to {base_filename}...")
    # Determine the data types from the first element
    x, first_y = data[0]
    first_int_val = x[2]
    dtype_int_val = np.array(first_int_val).dtype
    dtype_y = np.array(first_y).dtype

    chunk = data
    seq1s = [np.array(d[0][0]) for d in chunk]
    seq2s = [np.array(d[0][1]) for d in chunk]
    annotation = np.array([d[0][2] for d in chunk], dtype=dtype_int_val)
    y = np.array([d[1] for d in chunk], dtype=dtype_y)

    ##########################
    # Try pca
    ##########################
    seq1s = apply_pca_torch(np.array(seq1s),n_components=pca_components)
    seq2s = apply_pca_torch(np.array(seq2s),n_components=pca_components)

    # Save each component of the chunk
    filename = f'{base_filename}_seq1_{base_index}.npy'
    with open(filename, 'wb') as f:
        element = np.array(seq1s, dtype=np.float32)
        print(f"shape of seq1s: {element.shape}")
        print(f"first 10 elements of seq1s: {element[0][0][0][:10]}")
        data_to_save = element.tobytes()
        f.write(data_to_save)
    filename = f'{base_filename}_seq2_{base_index}.npy'
    with open(filename, 'wb') as f:
        data_to_save = np.array(seq2s, dtype=np.float32).tobytes()
        f.write(data_to_save)
    filename = f'{base_filename}_annot_{base_index}.npy'
    with open(filename, 'wb') as f:
        data_to_save = np.array(annotation, dtype=dtype_int_val).tobytes()
        f.write(data_to_save)
    filename = f'{base_filename}_y_{base_index}.npy'
    with open(filename, 'wb') as f:
        data_to_save = np.array(y, dtype=dtype_y).tobytes()
        f.write(data_to_save)

    seq1_paths = f'{base_filename}_seq1_{base_index}.npy'
    seq2_paths = f'{base_filename}_seq2_{base_index}.npy'
    annot_paths = f'{base_filename}_annot_{base_index}.npy'
    label_paths = f'{base_filename}_y_{base_index}.npy'
    print(f"Dtype of annotation: {dtype_int_val} and dtype of y: {dtype_y}")
    return seq1_paths, seq2_paths, annot_paths, label_paths


def load_data(seq1_paths, seq2_paths, annot_paths, label_paths):
    seq1 = np.load(seq1_paths, mmap_mode='r')
    seq2 = np.load(seq2_paths, mmap_mode='r')
    annot = np.load(annot_paths, mmap_mode='r')
    y = np.load(label_paths, mmap_mode='r')
    return seq1, seq2, annot, y
