import numpy as np
import math
def obj_func(x: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):
    return 0.5 * np.sum((A @ x - b)**2) + mu * np.sum(np.linalg.norm(x, axis=1))

def compute_nonzero_ratio(x, tol_factor=1e-6):
    x_flat = np.abs(x).ravel()
    max_abs = np.max(x_flat)
    
    if max_abs == 0:
        return 0.0
    
    threshold = tol_factor * max_abs
    num_nonzero = np.sum(x_flat > threshold)
    total = x_flat.size
    return num_nonzero / total

def cos_annealing(iter, max_iter, dt_min, dt_max):
    
    iter_cos_decay = round(max_iter)
    if iter >= iter_cos_decay:
        return dt_min
    else:
        return dt_min + (1 + math.cos(math.pi * (iter/iter_cos_decay) ) ) * (dt_max-dt_min) /2
    
# def result_struct_init_():
#     result = {
#         'f_values' : [], # values of object function
#         'sparsities': [],
#         ''
#     }
#     return result

