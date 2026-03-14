import numpy as np
from sklearn.decomposition import PCA

def get_subspace_basis(activations: np.ndarray, k: int) -> np.ndarray:
    centered_acts = activations - np.mean(activations, axis=0)
    
    pca = PCA(n_components=k)
    pca.fit(centered_acts)
    
    # pca.components_ is shape (k, d_model). Use transpose to get basis columns.
    return pca.components_.T

def subspace_overlap(basis_A: np.ndarray, basis_B: np.ndarray) -> float:
    # basis_A shape: (d_model, k_A) // basis_B shape: (d_model, k_B)
    d_min = min(basis_A.shape[1], basis_B.shape[1])
    
    # Interaction matrix M = Q_A^T @ Q_B
    M = basis_A.T @ basis_B
    
    # Singular values of M are the cosines of the principal angles
    squared_frobenius = np.sum(M ** 2)
    
    overlap_score = squared_frobenius / d_min
    return overlap_score

def compare_all_processes(activations: np.ndarray, labels: np.ndarray, k_dims: int = 2):
    unique_processes = np.unique(labels)
    num_processes = len(unique_processes)
    
    bases = {}
    for p_id in unique_processes:
        p_acts = activations[labels == p_id]
        bases[p_id] = get_subspace_basis(p_acts, k=k_dims)
        
    overlap_matrix = np.zeros((num_processes, num_processes))
    for i in range(num_processes):
        for j in range(num_processes):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                overlap_matrix[i, j] = subspace_overlap(bases[unique_processes[i]], bases[unique_processes[j]])
                
    return overlap_matrix