import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def one_hot_to_sequence(one_hot_array):
    """
    Convert an array of one-hot encoded DNA sequences back to their original sequence format.

    Parameters:
    - one_hot_array: NumPy array of shape (num_sequences, sequence_length, 4),
                     one-hot encoded sequences.

    Returns:
    - List of sequence strings.
    """
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    
    for seq in one_hot_array:
        sequence = ''.join([mapping[np.argmax(pos)] for pos in seq])
        sequences.append(sequence)
    
    return sequences

file_path = '../../../root/data/alternative_splicing/delta_logit.h5'
with h5py.File(file_path, 'r') as hdf:
    # List all groups
    print("Keys: %s" % hdf.keys())
    a_group_key = list(hdf.keys())

    print(a_group_key)

sequences = []
for key in a_group_key:
    if key.startswith('x_'):
        with h5py.File(file_path, 'r') as f:
            split_sequences = one_hot_to_sequence(np.array(f[key]))
            sequences = sequences + split_sequences
    if key.startswith('y_'):
        with h5py.File(file_path, 'r') as f:
            print(np.array(f[key]))
    if key.startswith('m_'):
        with h5py.File(file_path, 'r') as f:
            print(f[key])


