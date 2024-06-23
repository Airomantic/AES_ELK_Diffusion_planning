import numpy as np
import math
import cvxpy as cp
from kinematic_bicycle_model import VehicleInfo, update_ABMatrix


# 系统配置
NX = 3  # 状态向量的个数: x = [x, y, yaw]
NU = 2  # 输入向量的个数: u = [v, delta]
NP = 5  # 有限时间视界长度：预测过程中考虑的时间范围的有限长度
MAX_V = 20  # 最大车速(m/s)

# MPC config
Q = np.diag([2.0, 2.0, 2.0])  # 运行状态代价
F = np.diag([2.0, 2.0, 2.0])  # 末端状态代价
R = np.diag([0.01, 0.1])  # 输入状态代价


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

    xref = np.zeros((NX, NP + 1))
    uref = np.zeros((NU, NP + 1))

    for i in range(NP+1):
        ind = min(index + i, len(rx)-1)
        xref[0, i] = rx[ind]
        xref[1, i] = ry[ind]
        xref[2, i] = ryaw[ind]
        uref[0, i] = rv[ind]
        uref[1, i] = math.atan2(vehicle.L*rkappa[ind], 1)
    return xref, uref, index, er



def MPCController(vehicle, ref_path):
    xref, uref, index, er = calc_preparation(vehicle, ref_path)

    x0 = [vehicle.x, vehicle.y, vehicle.yaw]
    x = cp.Variable((NX, NP + 1))
    u = cp.Variable((NU, NP))
    cost = 0.0  # 代价函数
    constraints = []  # 约束条件

    x[:, 0] == x0
    for i in range(NP):
        cost += cp.quad_form(u[:, i] - uref[:, i], R)
        if i != 0:
            cost += cp.quad_form(x[:, i] - xref[:, i], Q)
        A, B = update_ABMatrix(vehicle, uref[1, i], xref[2, i])
        constraints += [x[:, i + 1] - xref[:, i + 1] == A @ (x[:, i] - xref[:, i]) + B @ (u[:, i] - uref[:, i])]

    cost += cp.quad_form(x[:, NP] - xref[:, NP], F)

    constraints += [(x[:, 0]) == x0]
    constraints += [cp.abs(u[0, :]) <= MAX_V]
    constraints += [cp.abs(u[1, :]) <= VehicleInfo.MAX_STEER]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        opt_delta = u.value[:, 0]
        return opt_delta[1], index, er

    else:
        print("Error: MPC solution failed !")
        return None, index, er