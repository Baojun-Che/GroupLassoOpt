import numpy as np
import matplotlib.pyplot as plt
import os
from basic_setting import SGD_hyperparams, compute_group_sparsity

def subgrad_regular(x):
    n, l = x.shape
    grad = np.zeros_like(x)
    for i in range(n):
        row_i = x[i, :]
        norm_row_i = np.linalg.norm(row_i)
        if norm_row_i > 1e-6:
            grad[i, :] = row_i / norm_row_i
        else:
            grad[i, :] = row_i / (norm_row_i + 1)
    return grad

def gl_SGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):

    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    
    max_iter = 10000
    dt_max = 1e-4
    alpha = 0.001
    tol = 0.01
    window_size = 1000
    
    f_values = []
    
    obj = 0.5 * np.sum((A @ x - b)**2) + mu * np.sum(np.linalg.norm(x, axis=1))
    f_values.append(obj)
    
    x_best = x.copy()
    best_obj = obj
    
    iter_count = 0
    
    for k in range(max_iter):

        if (k+1) % (max_iter/10) == 0:
            print(f"Running {k+1} iteration")
        d = A @ x - b
        grad_smooth = A.T @ d
        subgrad_non_smooth = mu * subgrad_regular(x)
        subgrad_total = grad_smooth + subgrad_non_smooth 
        
        dt = min(dt_max, alpha / np.sqrt(k+1))
        x = x - dt * subgrad_total 
        
        obj = 0.5 * np.sum(d**2) + mu * np.sum(np.linalg.norm(x, axis=1))
        f_values.append(obj)
        
        if obj < best_obj:
            best_obj = obj
            x_best = x.copy()
        
        iter_count += 1
        
        if len(f_values) >= window_size:
            recent = f_values[-window_size:]
            if max(recent) - min(recent) <= tol:
                break
    
    return x_best, iter_count, f_values



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
    x_best, iter_count, f_values = gl_SGD_primal(x0, A, b, mu)
    
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {min(f_values):.6f}")
    print(f"解的稀疏度: {compute_group_sparsity(x_best)}")

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
