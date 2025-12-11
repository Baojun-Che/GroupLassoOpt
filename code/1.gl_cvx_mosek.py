import numpy as np
import cvxpy as cp
from utility import compute_nonzero_ratio

def gl_CVX_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):
    
    m, n = A.shape
    _, l = b.shape

    x = cp.Variable((n, l))

    # Build group norm
    group_norm = 0
    for i in range(n):
        group_norm += cp.norm(x[i, :], 2)

    # Objective function
    obj = 0.5 * cp.sum_squares(A @ x - b) + mu * group_norm

    # Problem
    prob = cp.Problem(cp.Minimize(obj))

    # Solve with MOSEK
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        raise RuntimeError(f"MOSEK failed to solve the problem: {e}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Problem not solved to optimality. Status: {prob.status}")

    x_opt = x.value

    residual = A @ x_opt - b
    frob_sq = 0.5 * np.sum(residual ** 2)
    group_sum = np.sum(np.linalg.norm(x_opt, axis=1))
    f_final = frob_sq + mu * group_sum

    f_values = [f_final]

    return x_opt, -1, f_values



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
    x_opt, iter_count, f_values = gl_CVX_mosek(x0, A, b, mu)
    
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {min(f_values):.6f}")
    print(f"解的稀疏度: {compute_nonzero_ratio(x_opt)}")

    np.save("code/solutions/opt_mosek.npy", x_opt)
