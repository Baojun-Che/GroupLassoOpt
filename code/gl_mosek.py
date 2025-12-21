import numpy as np
import mosek
import utils, time
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix

def gl_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float):
    """
    使用 MOSEK Fusion API 求解 Group LASSO 问题
    min_{x} 0.5 * ||A x - b||_F^2 + mu * sum_i ||x(i,:)||_2
    
    参数:
        x0: 初始解（被忽略，仅用于接口一致性）
        A: 矩阵 A (m x n)
        b: 矩阵 b (m x l)
        mu: 正则化参数
    
    返回:
        [x_opt, -1, [opt_value]] 最优解、迭代次数（固定为-1）和最优值
    """
    m, n = A.shape
    l = b.shape[1]
    
    # 创建 MOSEK Fusion 模型
    with Model('GroupLASSO_MOSEK') as M:
        # 关闭日志输出
        M.setLogHandler(None)
        
        # 将 numpy 数组转换为 MOSEK 矩阵
        A_mat = Matrix.dense(A)
        b_mat = Matrix.dense(b)
        
        # 变量 X (n x l)
        X = M.variable("X", [n, l], Domain.unbounded())
        
        # 变量 Y (m x l) 用于残差
        Y = M.variable("Y", [m, l], Domain.unbounded())
        
        # 变量 t0 用于 ||Y||_F^2 的旋转锥表示
        t0 = M.variable(1, Domain.unbounded())
        
        # 变量 ts (n) 用于组范数
        ts = M.variable(n, Domain.greaterThan(0.0))
        
        # 约束: Y = A X - b
        M.constraint(
            Expr.sub(Expr.sub(Expr.mul(A_mat, X), b_mat), Y),
            Domain.equalsTo(0.0)
        )
        
        # 旋转二次锥约束: t0 >= 0.5 * ||Y||_F^2
        # 等价于: (t0, 1, vec(Y)) 在旋转二次锥中
        Y_vec = Y.reshape(m * l)
        M.constraint(
            Expr.vstack(t0, Expr.constTerm(1, 1.0), Y_vec),
            Domain.inRotatedQCone()
        )
        
        # 二阶锥约束: ts_i >= ||X[i,:]||_2
        # 需要将 X 的行转换为向量
        for i in range(n):
            # 获取 X 的第 i 行，然后转换为向量
            X_row = X.slice([i, 0], [i+1, l])
            X_row_vec = Expr.reshape(X_row, l)
            
            # 二阶锥约束: (ts_i, X_row) 在二次锥中
            M.constraint(
                Expr.vstack(ts.index(i), X_row_vec),
                Domain.inQCone()
            )
        
        # 目标函数: t0 + mu * sum(ts)
        obj = Expr.add(t0, Expr.mul(mu, Expr.sum(ts)))
        M.objective('obj', ObjectiveSense.Minimize, obj)
        
        # 求解
        M.solve()
        
        # 获取解
        x_opt = X.level().reshape(n, l)
        opt_value = M.primalObjValue()
        
        return x_opt, -1, [opt_value]


if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_mosek(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {utils.compute_nonzero_ratio(x_opt)}")
