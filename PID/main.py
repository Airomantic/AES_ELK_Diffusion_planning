import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.spatial import KDTree
from bicycle_model import Vehicle, VehicleInfo, draw_trailer
from PID import PIDController
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio

# https://mp.weixin.qq.com/s?__biz=MzI4ODc4ODM1MA==&mid=2247485715&idx=1&sn=e6596fe172dbb1b965489382f43da833&scene=21#wechat_redirect
MAX_SIMULATION_TIME = 200.0  # 程序最大运行时间200*dt
PID = PIDController(2, 0.001, 3)

def NormalizeAngle(angle):
    a = math.fmod(angle + np.pi, 2 * np.pi)
    if a < 0.0:
        a += (2.0 * np.pi)
    return a - np.pi


def main():
    # 设置跟踪轨迹
    ref_path = np.zeros((1000, 2))
    ref_path[:, 0] = np.linspace(0, 50, 1000)  # x坐标
    ref_path[:, 1] = 10 * np.sin(ref_path[:, 0] / 20.0)  # y坐标
    ref_tree = KDTree(ref_path)
    # 假设车辆初始位置为（0，0），航向角yaw=0.0，速度为2m/s，时间周期dt为0.1秒
    vehicle = Vehicle(x=0.0,
                      y=0.0,
                      yaw=np.pi/2,
                      v=2.0,
                      dt=0.1,
                      l=VehicleInfo.L)

    time = 0.0  # 初始时间

    # 记录车辆轨迹
    trajectory_x = []
    trajectory_y = []
    lat_err = []  # 记录横向误差

    i = 0
    image_list = []  # 存储图片
    plt.figure(1)

    last_idx = ref_path.shape[0] - 1  # 跟踪轨迹的最后一个点的索引
    old_idx = 0  # 记录上一次的索引点
    target_ind = 0  # 第一个目标点的索引
    while MAX_SIMULATION_TIME >= time and last_idx > target_ind:
        time += vehicle.dt  # 累加一次时间周期
        vehicle_positon = np.zeros(2)
        vehicle_positon[0] = vehicle.x
        vehicle_positon[1] = vehicle.y
        distance, target_ind = ref_tree.query(vehicle_positon)  # 在跟踪轨迹上查找离车辆最近的点
        if old_idx > target_ind:
            print("ERROR: Find the point behind the vehicle.")
            target_ind = old_idx + 1  # 查找到车辆后面的点，将目标点索引置为上一次的索引点idx+1
        old_idx = target_ind  # 记录本次索引点idx
        alpha = math.atan2(
            ref_path[target_ind, 1] - vehicle_positon[1], ref_path[target_ind, 0] - vehicle_positon[0])
        l_d = np.linalg.norm(ref_path[target_ind] - vehicle_positon)  # 目标点与车定位点距离l_d

        theta_e = NormalizeAngle(alpha - vehicle.yaw)
        e_y = l_d * math.sin(theta_e)  # 计算实际误差，0为目标距离
        lat_err.append(e_y)
        delta_f = PID.cal_output(e_y, vehicle.dt)

        vehicle.update(0.0, delta_f, np.pi / 10)  # 由于假设纵向匀速运动，所以加速度a=0.0
        trajectory_x.append(vehicle.x)
        trajectory_y.append(vehicle.y)

        # 显示动图
        plt.cla()
        plt.plot(ref_path[:, 0], ref_path[:, 1], '-.b', linewidth=1.0)
        draw_trailer(vehicle.x, vehicle.y, vehicle.yaw, vehicle.steer, plt)

        plt.plot(trajectory_x, trajectory_y, "-r", label="trajectory")
        plt.plot(ref_path[target_ind, 0], ref_path[target_ind, 1], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        plt.savefig("temp.png")
        i += 1
        if (i % 50) > 0:
            image_list.append(imageio.imread("temp.png"))

    imageio.mimsave("display.gif", image_list, duration=0.1)

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(ref_path[:, 0], ref_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(trajectory_x, trajectory_y, 'r')
    plt.title("actual tracking effect")

    plt.subplot(2, 1, 2)
    plt.plot(lat_err)
    plt.title("lateral error")
    plt.show()


if __name__ == '__main__':
    main()
