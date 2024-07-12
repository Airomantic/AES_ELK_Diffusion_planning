import numpy as np
import math
import cvxpy as cp
from kinematic_bicycle_model import VehicleInfo, update_ABMatrix

"""
QP矩阵求解方法和目的
MPC通过求解一个二次规划（Quadratic Programming, QP）问题来找到最优控制输入。QP问题是一种特殊类型的优化问题，其目标函数是二次的，约束条件可以是线性的。

方法：这里使用CVXPY库，它是一个基于Python的凸优化问题求解器，可以方便地定义和求解QP问题。
目的：MPC的目的是通过预测未来的状态和控制输入，找到一条最优的控制轨迹，使得车辆能够以最小的偏差和代价跟踪给定的参考轨迹。
在MPC中，QP问题通常在每个控制周期重复求解，以适应车辆状态的变化和可能的外部扰动。这种方法可以提供良好的跟踪性能和鲁棒性。
"""

# 系统配置
NX = 3  # 车辆的状态，状态向量的个数: x = [x, y, yaw]
NU = 2  # 控制，输入向量的个数: u = [v, delta] 速度（v）和转向角（delta）
NP = 5  # 有限时间视界长度：预测过程中考虑的时间范围的有限长度
MAX_V = 20  # 最大车速(m/s)

# MPC config
Q = np.diag([2.0, 2.0, 2.0])  # 运行状态代价 ： 用于惩罚状态变量的偏差。
F = np.diag([2.0, 2.0, 2.0])  # 末端状态代价 ： 用于惩罚末端状态的偏差。
R = np.diag([0.01, 0.1])  # 输入状态代价 ： 用于惩罚控制输入的偏差。


def calc_preparation(vehicle, ref_path):
    """
    计算xref、uref、index和er
    """

    rx, ry, rv, ryaw, rkappa = ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], ref_path[:, 3], ref_path[:, 5]
    dx = [vehicle.x - icx for icx in rx]
    dy = [vehicle.y - icy for icy in ry]
    d = np.hypot(dx, dy)
    index = np.argmin(d)

    vec_nr = np.array([math.cos(ryaw[index] + math.pi / 2.0),
                       math.sin(ryaw[index] + math.pi / 2.0)])
    vec_target_2_rear = np.array([vehicle.x - rx[index],
                                  vehicle.y - ry[index]])
    er = np.dot(vec_target_2_rear, vec_nr)
    # 初始化两个矩阵
    xref = np.zeros((NX, NP + 1)) # 行数NX对应于状态向量的维度，列数NP + 1对应于预测时间步的数量（包括当前时刻）。每个元素xref[i, j]代表在第j个时间步，对应于第i个状态变量的参考值
    uref = np.zeros((NU, NP + 1)) # 行数NU对应于控制输入向量的维度，列数同样为NP + 1

    for i in range(NP+1):
        ind = min(index + i, len(rx)-1)
        xref[0, i] = rx[ind]
        xref[1, i] = ry[ind]
        xref[2, i] = ryaw[ind]
        uref[0, i] = rv[ind]
        uref[1, i] = math.atan2(vehicle.L*rkappa[ind], 1)
    return xref, uref, index, er


# QP矩阵求解
def MPCController(vehicle, ref_path):
    xref, uref, index, er = calc_preparation(vehicle, ref_path)

    x0 = [vehicle.x, vehicle.y, vehicle.yaw]
    x = cp.Variable((NX, NP + 1))  # 表示在未来NP + 1个时间步的状态预测
    u = cp.Variable((NU, NP))     # 表示在NP个时间步的控制输入, 其中 u[0, :] 表示所有时间步的速度，u[1, :] 表示所有时间步的转向角。
    cost = 0.0  # 代价函数   
    constraints = []  # 约束条件
    
    # 目标函数通常包括对状态变量 x 和控制输入 u 的惩罚项，以最小化预测轨迹与参考轨迹之间的偏差。约束条件则确保这些变量满足系统的动力学模型和物理限制。

    # quad_form用于计算一个向量与一个正定矩阵的乘积的二次形式
    x[:, 0] == x0
    for i in range(NP):
        # 计算的是向量 x 与矩阵 P 的乘积的二次形式，即 x^T * P * x
        # u[:, i] - uref[:, i]：计算当前控制输入与参考控制输入之间的偏差
        # R：是输入代价矩阵，通常是一个对角矩阵，用于对控制输入的偏差进行加权
        # 将这个值加到 cost 变量上，意味着在优化问题的目标函数中加入了对控制输入偏差的惩罚项。这样做的目的是使得求解的控制输入尽可能接近参考控制输入，从而保证车辆的控制平滑性和跟踪精度。
        cost += cp.quad_form(u[:, i] - uref[:, i], R) 
        if i != 0:
            # 目的是使得优化算法寻找的控制序列不仅使得控制输入平滑，而且确保状态向量的预测值尽可能接近参考轨迹
            cost += cp.quad_form(x[:, i] - xref[:, i], Q)
        # 系统动态模型：更新系统动态矩阵A和B，这些矩阵描述了车辆状态如何根据控制输入变化。
        A, B = update_ABMatrix(vehicle, uref[1, i], xref[2, i])
        # 构建系统动力学模型的约束条件
        """
        将这个等式作为约束条件添加到优化问题中，意味着我们要求优化算法找到的控制序列 u 和状态序列 x 必须满足系统的动力学模型。
        换句话说，我们希望车辆的实际运动遵循物理规律，即给定当前状态和控制输入，下一状态的预测必须与系统动态一致。
        """
        constraints += [x[:, i + 1] - xref[:, i + 1] == A @ (x[:, i] - xref[:, i]) + B @ (u[:, i] - uref[:, i])]

    cost += cp.quad_form(x[:, NP] - xref[:, NP], F) # 包括输入偏差的代价、状态偏差的代价和末端状态偏差的代价

    # 设置约束条件
    constraints += [(x[:, 0]) == x0] # 初始状态约束：确保优化开始时车辆的状态与当前状态一致
    constraints += [cp.abs(u[0, :]) <= MAX_V]  # 输入和状态的物理约束：例如速度不能超过MAX_V，转向角不能超过车辆的最大转向能力
    constraints += [cp.abs(u[1, :]) <= VehicleInfo.MAX_STEER]

    # 求解优化问题
    problem = cp.Problem(cp.Minimize(cost), constraints) # CVXPY库中的Problem类来定义并求解优化问题
    problem.solve(solver=cp.ECOS, verbose=False)         # ECOS求解器，并且设置为不输出详细信息。

    # 求解结果
    # 如果求解成功，返回计算出的最优转向角opt_delta[1]，最近点的索引index和横向误差er。
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        opt_delta = u.value[:, 0]
        return opt_delta[1], index, er

    else:
        print("Error: MPC solution failed !")
        return None, index, er