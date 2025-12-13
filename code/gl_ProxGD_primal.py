import math
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from utility import plot_relative_error, compute_nonzero_ratio

def prox_group_lasso(z, dt, mu):

    norms = np.linalg.norm(z, axis=1, keepdims=True)  # Shape: (n, 1)
    k = dt * mu
    scaling = np.maximum(0.0, 1.0 - k / (norms + 1e-8))      
    x_next = scaling * z
    
    return x_next

def gl_ProxGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):
    
    norm_A = np.linalg.norm(A, ord=2)
    Lip =  norm_A**2 + 1e3 
    
    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    
    max_iter_total = 5000
    max_iter_inner = 500
    dt = 1.0/Lip
    tol = 1e-3
    
    f_values = []
    
    x_opt = np.zeros_like(x)
    best_obj = 0.5 * np.sum(b**2)
    
    iter_count = 0
    mu_current = 2e4 * mu
    flag = False

    while(1):

        mu_current = 0.1 * mu_current
        tol = 0.1* tol

        obj_current_old = 0.0

        if mu_current <=mu:
            mu_current = mu
            tol = 1e-7
            flag = True
            max_iter_inner = 1000

        print(f"Iterations: {iter_count}, current mu={mu_current}")

        for k in range(max_iter_inner):

            r = A @ x - b
            f_smooth = 0.5 * np.sum(r**2)
            f_regular = np.sum(np.linalg.norm(x, axis=1))

            obj = f_smooth + mu * f_regular
            f_values.append(obj)
            iter_count += 1
            if obj < best_obj:
                x_opt = x
                best_obj = obj
                
            obj_current_new = f_smooth + mu_current * f_regular
            if np.abs(obj_current_new - obj_current_old) < tol:
                break

            obj_current_old = obj_current_new

            grad_smooth = A.T @ r
            z = x - dt * grad_smooth
            x = prox_group_lasso(z, dt, mu_current)

            if iter_count >= max_iter_total:
                flag = True
                break
        
        if flag :
            break

    
    return x_opt, iter_count, f_values



if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_ProxGD_primal(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {compute_nonzero_ratio(x_opt)}")

    plot_relative_error(f_values, "doc/figs/PGD", 0.6705752210556729)