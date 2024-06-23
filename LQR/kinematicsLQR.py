import numpy as np
import math
from scipy.linalg import inv
from kinematic_bicycle_model import update_ABMatrix


N = 200  # 迭代范围
EPS = 1e-4  # 迭代精度
Q = np.eye(3) * 8
R = np.eye(2) * 2
F = np.eye(3) * 10


def cal_lqr_k(A, B, Q, R, F):
    """计算LQR反馈矩阵K
    Args:
        A : mxm状态矩阵A
        B : mxn状态矩阵B
        Q : Q是状态权重mxm的半正定方阵，用于衡量状态的代价，通常将其设计为对角矩阵。
        R : R是控制权重nxn的正定对称矩阵，用于衡量控制输入的代价，通常将其设计为对角矩阵。
        F : F是末端状态权重mxm的半正定方阵，用于衡量最终状态的代价，通常将其设计为对角矩阵。
    Returns:
        K : 反馈矩阵K
    """
    # 设置迭代初始值
    P = F
    # 循环迭代
    for t in range(N):
        K_t = inv(B.T @ P @ B + R) @ B.T @ P @ A
        P_t = (A - B @ K_t).T @ P @ (A - B @ K_t) + Q + K_t.T @ R @ K_t
        if (abs(P_t - P).max() < EPS):
            break
        P = P_t
    return K_t


def normalize_angle(angle):
    a = math.fmod(angle + np.pi, 2 * np.pi)
    if a < 0.0:
        a += (2.0 * np.pi)
    return a - np.pi


def calc_preparation(vehicle, ref_path):
    """
    计算角度误差theta_e、横向误差er、曲率rk和索引index
    """

    rx, ry, ref_yaw, ref_kappa = ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], ref_path[:, 4]
    dx = [vehicle.x - icx for icx in rx]
    dy = [vehicle.y - icy for icy in ry]
    d = np.hypot(dx, dy)
    index = np.argmin(d)
    rk = ref_kappa[index]
    ryaw = ref_yaw[index]
    rdelta = math.atan2(vehicle.L * rk, 1)

    vec_nr = np.array([math.cos(ryaw + math.pi / 2.0),
                       math.sin(ryaw + math.pi / 2.0)])

    vec_target_2_rear = np.array([vehicle.x - rx[index],
                                  vehicle.y - ry[index]])

    er = np.dot(vec_target_2_rear, vec_nr)
    theta_e = normalize_angle(vehicle.yaw - ryaw)

    return dx[index], dy[index], theta_e, er, rdelta, ryaw, index


def LQRController(vehicle, ref_path):
    x_e, y_e, theta_e, er, rdelta, ryaw, index = calc_preparation(vehicle, ref_path)
    x = np.matrix([[x_e],
                   [y_e],
                   [theta_e]])
    A, B = update_ABMatrix(vehicle, rdelta, ryaw)
    K = cal_lqr_k(A, B, Q, R, F)

    u = -K @ x
    delta_f = rdelta + u[1,0]
    return delta_f, index, er