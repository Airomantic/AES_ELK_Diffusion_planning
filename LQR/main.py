from kinematic_bicycle_model import Vehicle, VehicleInfo, draw_vehicle
from kinematicsLQR import LQRController
from path_generator import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

MAX_SIMULATION_TIME = 200.0  # 程序最大运行时间200*dt

def main():
    # 设置跟踪轨迹
    rx, ry, ref_yaw, ref_s, ref_kappa = Path().get_ref_line_info()
    ref_path = np.column_stack((rx, ry, ref_yaw, ref_s, ref_kappa))
    # 假设车辆初始位置为（5，60），航向角yaw=0.0，速度为2m/s，时间周期dt为0.1秒
    vehicle = Vehicle(x=5.0,
                      y=60.0,
                      yaw=0.0,
                      v=2.0,
                      dt=0.1,
                      l=VehicleInfo.L)

    time = 0.0  # 初始时间
    target_ind = 0
    # 记录车辆轨迹
    trajectory_x = []
    trajectory_y = []
    lat_err = []  # 记录横向误差

    i = 0
    image_list = []  # 存储图片
    plt.figure(1)

    last_idx = ref_path.shape[0] - 1  # 跟踪轨迹的最后一个点的索引
    while MAX_SIMULATION_TIME >= time and last_idx > target_ind:
        time += vehicle.dt  # 累加一次时间周期

        # rear_wheel_feedback
        delta_f, target_ind, e_y = LQRController(vehicle, ref_path)

        # 横向误差
        lat_err.append(e_y)

        # 更新车辆状态
        vehicle.update(0.0, delta_f, np.pi / 10)  # 由于假设纵向匀速运动，所以加速度a=0.0
        trajectory_x.append(vehicle.x)
        trajectory_y.append(vehicle.y)

        # 显示动图
        plt.cla()
        plt.plot(ref_path[:, 0], ref_path[:, 1], '-.b', linewidth=1.0)
        draw_vehicle(vehicle.x, vehicle.y, vehicle.yaw, vehicle.steer, plt)

        plt.plot(trajectory_x, trajectory_y, "-r", label="trajectory")
        plt.plot(ref_path[target_ind, 0], ref_path[target_ind, 1], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        plt.title("LQR trajectory")
        plt.savefig("picture_result/temp.png")
        i += 1
        if (i % 5) == 0:
            print(imageio.imread("picture_result/temp.png").shape)
            image_list.append(imageio.imread("picture_result/temp.png"))
    
    imageio.mimsave("picture_result/LQR_display.gif", image_list, duration=0.1)

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(ref_path[:, 0], ref_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(trajectory_x, trajectory_y, 'r')
    plt.title("LQR actual tracking effect")

    plt.subplot(2, 1, 2)
    plt.plot(lat_err)
    plt.title("lateral error")
    plt.savefig("picture_result/LQR_lateral_error.png")
    plt.show()


if __name__ == '__main__':
    main()
