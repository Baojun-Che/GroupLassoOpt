import numpy as np
import matplotlib.pyplot as plt
import os
import time
from utils import compute_nonzero_ratio, cos_annealing

def subgrad_regular(x):
    n, l = x.shape
    grad = np.zeros_like(x)
    for i in range(n):
        row_i = x[i, :]
        norm_row_i = np.linalg.norm(row_i)
        if norm_row_i > 1e-6:
            grad[i, :] = row_i / norm_row_i
        else:
            grad[i, :] = row_i / (1 + norm_row_i)
    return grad

def gl_SGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):

    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    
    N_out_iter = 10
    max_iter_total = 10000
    max_iter_inner = 500
    alpha = 0.5
    dt_max = 0.001
    tol = 1e-3
    
    f_values = []
    
    x_opt = np.zeros_like(x)
    best_obj = 0.5 * np.sum(b**2)
    
    flag = False

    for out_iter in range(N_out_iter):

        mu_current = cos_annealing(out_iter, N_out_iter - 1, mu, max(2.0, mu))

        obj_current_old = 0.0

        if out_iter == N_out_iter - 1:
            flag = True
            tol = 1e-7
            max_iter_inner = 5000

        for k in range(max_iter_inner):

            r = A @ x - b
            f_smooth = 0.5 * np.sum(r**2)
            f_regular = np.sum(np.linalg.norm(x, axis=1))

            obj = f_smooth + mu * f_regular
            f_values.append(obj)
            if obj < best_obj:
                x_opt = x
                best_obj = obj
                
            obj_current_new = f_smooth + mu_current * f_regular
            if np.abs(obj_current_new - obj_current_old) < tol:
                break

            obj_current_old = obj_current_new

            grad_smooth = A.T @ r
            subgrad_non_smooth = mu_current * subgrad_regular(x)
            subgrad_total = grad_smooth + subgrad_non_smooth 
            
            dt = min( dt_max, alpha / (k+1) )

            x = x - dt * subgrad_total
            if len(f_values)-1 >= max_iter_total:
                flag = True
                break
        
        if flag :
            break

        tol = max(tol*0.8, 1e-6)
        dt_max *= 0.9      
    
    r = A @ x - b
    f_smooth = 0.5 * np.sum(r**2)
    f_regular = np.sum(np.linalg.norm(x, axis=1))

    obj = f_smooth + mu * f_regular
    f_values.append(obj)
    if obj < best_obj:
        x_opt = x
        best_obj = obj

    return x_opt, len(f_values)-1, f_values



if __name__ == "__main__":

    np.random.seed(97006855)

    n = 512
    m = 256
    k = round(n * 0.1)
    l = 2

    A = np.random.randn(m, n)

    # 生成索引 p：随机选择前 k 个位置
    p = np.random.permutation(n)[:k]

    # 初始化 u: n x l，只在 p 对应的位置有值
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)

    b = A @ u
    mu = 0.01

    print(f"目标函数的全局最小值应不大于: { mu * np.sum(np.linalg.norm(u, axis=1))}")

    ########## 测试SGD算法 ##########

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_SGD_primal(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {compute_nonzero_ratio(x_opt)}")

    plt.figure(figsize=(8, 6))
    plt.semilogy(f_values)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.grid(True)
    plt.tight_layout()
    
    output_dir = "doc/figs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "SGD.pdf"))
    plt.show()
