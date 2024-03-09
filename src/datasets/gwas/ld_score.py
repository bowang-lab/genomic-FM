import sgkit as sg
import numpy as np
import scipy.sparse as sp_sparse

def calculate_LD_score(variant_ds, radius=100000, threshold=0.1):
    """
    Calculate the LD Score for a given variant within a specified genomic window.

    Parameters:
    - variant_ds (xarray.Dataset): Dataset containing variant and genotype data.
    - radius (int): Radius around the variant position to calculate LD in base pairs.
    - threshold (float): Threshold for LD pruning.

    Returns:
    - float: The LD Score for the given variant.
    """
    # Ensure the radius is an integer
    radius = int(radius)

    ds["call_dosage"] = ds.call_genotype.sum(dim="ploidy")

    # Calculate LD matrix within the specified window
    ds = sg.window_by_position(
        variant_ds, size=radius, offset=-radius/2, window_start_position="variant_position",
    )
    ld = sg.ld_matrix(ds, threshold=threshold)
    ld = ld.compute()

    # Convert LD matrix to CSR format for efficient row sum calculations
    ld_csr = sp_sparse.coo_matrix((ld.values, (ld.variant_i.values, ld.variant_j.values))).tocsr()

    # Calculate the LD Score as the sum of the squared LD values for the variant
    ld_score = np.sum(ld_csr**2)

    return ld_score

def calculate_stratified_ld_score(ld_matrix, model_scores):
    """
    Calculate stratified LD Scores by multiplying the LD matrix with model scores.

    Parameters:
    - ld_matrix (scipy.sparse.csr_matrix): A sparse matrix representing LD between variants.
    - gpn_scores (numpy.ndarray): An array of model scores corresponding to each variant in the LD matrix.

    Returns:
    - numpy.ndarray: An array of stratified LD Scores for each variant.
    """

    if not isinstance(model_scores, np.ndarray):
        model_scores = np.array(model_scores)
    
    # Check if dimensions match
    if ld_matrix.shape[0] != model_scores.shape[0]:
        raise ValueError("The dimensions of the LD matrix and model scores do not match.")
    
    # Multiply LD matrix by model scores
    stratified_ld_scores = ld_matrix.dot(model_scores)
    
    # Take the absolute value if considering unsigned scores and LD
    stratified_ld_scores = np.abs(stratified_ld_scores)
    
    return stratified_ld_scores
