import numpy as np

def obj_func(x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):
    return 0.5 * np.sum((A @ x - b)**2) + mu * np.sum(np.linalg.norm(x, axis=1))

def compute_group_sparsity(x, tol=1e-3):
    
    row_norms = np.linalg.norm(x, axis=1)          
    zero_groups = row_norms < tol 
    num_zero_groups = np.sum(zero_groups)
    sparsity_ratio = num_zero_groups / x.shape[0]
    
    return sparsity_ratio

def SGD_hyperparams():
    pass

