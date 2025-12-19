import numpy as np
import matplotlib.pyplot as plt
import math
from gl_cvx_mosek import gl_cvx_mosek

def compute_nonzero_ratio(x, tol_factor=1e-6):
    x_flat = np.abs(x).ravel()
    max_abs = np.max(x_flat)
    
    if max_abs == 0:
        return 0.0
    
    threshold = tol_factor * max_abs
    num_nonzero = np.sum(x_flat > threshold)
    total = x_flat.size
    return num_nonzero / total

def prox_group_lasso(z, k):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scaling = np.maximum(0.0, 1.0 - k / (norms + 1e-8))
    return scaling * z

def cos_annealing(iter, max_iter, dt_min, dt_max):
    
    iter_cos_decay = round(max_iter)
    if iter >= iter_cos_decay:
        return dt_min
    else:
        return dt_min + (1 + math.cos(math.pi * (iter/iter_cos_decay) ) ) * (dt_max-dt_min) /2

def test_data_init(seed = 97006855, n = 512, m = 256, l = 2, mu = 0.01, sparse = 0.1, save_data = False):
    
    np.random.seed(seed)

    k = round(n * 0.1)
    A = np.random.randn(m, n)

    # 生成索引 p：随机选择前 k 个位置
    p = np.random.permutation(n)[:k]

    # 初始化 u: n x l，只在 p 对应的位置有值
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)

    b = A @ u

    print(f"目标函数的全局最小值应不大于: { mu * np.sum(np.linalg.norm(u, axis=1))}")

    ########## 测试CVX-mosek ##########

    x0 = np.zeros((n, l))
    x_opt, _, f_values = gl_cvx_mosek(x0, A, b, mu)
    
    print(f"CVX-mosek 求得目标函数最小值: {min(f_values):.6f}")
    print(f"解的稀疏度: {compute_nonzero_ratio(x_opt)}")

    if save_data:
        with open('code/datas/obj_opt.txt', 'w') as f:
            f.write(str(min(f_values)))

        np.save("code/datas/opt_mosek.npy", x_opt)
        np.save("code/datas/A.npy", A)
        np.save("code/datas/b.npy", b)
        np.save("code/datas/u.npy", u)
    
    return A, b, u

def plot_relative_error(f_values, fig_name, obj_opt = -1):

    if obj_opt < 0:
        with open('obj_opt.txt', 'r') as f:
            obj_opt = float(f.read().strip())

    assert obj_opt>0

    f_values = np.array(f_values)
    
    rel_err = (f_values - obj_opt) / obj_opt
    rel_err = np.maximum(rel_err, 1e-16)
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(rel_err, linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('doc/figs/'+ fig_name + '.pdf', format='pdf')
    plt.close()

def recover_primal_solution(Z: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, eps = 1e-6):

    m, n = A.shape
    _, l = b.shape
    active = np.linalg.norm(Z, axis=1) >= mu - 1e-6
    x = np.zeros((n, l))

    if np.any(active):
        A_active = A[:, active]  # m x n_active
        x_active_part = np.linalg.lstsq(A_active, b, rcond=None)[0]
        x[active, :] = x_active_part
    return x

if __name__ == "__main__":
    test_data_init()