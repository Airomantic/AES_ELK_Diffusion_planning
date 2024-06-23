"""
路径轨迹生成器
"""

import math
import numpy as np

class Path:
    def __init__(self):
        self.ref_line = self.design_reference_line()
        self.ref_yaw = self.cal_yaw()
        self.ref_s = self.cal_accumulated_s()
        self.ref_kappa = self.cal_kappa()

    def design_reference_line(self):
        rx = np.linspace(0, 50, 2000) + 5 # x坐标
        ry = 20 * np.sin(rx / 20.0) + 60  # y坐标
        rv = np.full(2000, 2)
        return np.column_stack((rx, ry, rv))
    def cal_yaw(self):
        yaw = []
        for i in range(len(self.ref_line)):
            if i == 0:
                yaw.append(math.atan2(self.ref_line[i + 1, 1] - self.ref_line[i, 1],
                                 self.ref_line[i + 1, 0] - self.ref_line[i, 0]))
            elif i == len(self.ref_line) - 1:
                yaw.append(math.atan2(self.ref_line[i, 1] - self.ref_line[i - 1, 1],
                                 self.ref_line[i, 0] - self.ref_line[i - 1, 0]))
            else:
                yaw.append(math.atan2(self.ref_line[i + 1, 1] - self.ref_line[i -1, 1],
                                  self.ref_line[i + 1, 0] - self.ref_line[i - 1, 0]))
        return yaw

    def cal_accumulated_s(self):
        s = []
        for i in range(len(self.ref_line)):
            if i == 0:
                s.append(0.0)
            else:
                s.append(math.sqrt((self.ref_line[i, 0] - self.ref_line[i-1, 0]) ** 2
                                 + (self.ref_line[i, 1] - self.ref_line[i-1, 1]) ** 2))
        return s

    def cal_kappa(self):
        # 计算曲线各点的切向量
        dp = np.gradient(self.ref_line.T, axis=1)
        # 计算曲线各点的二阶导数
        d2p = np.gradient(dp, axis=1)
        # 计算曲率
        kappa = (d2p[0] * dp[1] - d2p[1] * dp[0]) / ((dp[0] ** 2 + dp[1] ** 2) ** (3 / 2))

        return kappa

    def get_ref_line_info(self):
        return self.ref_line[:, 0], self.ref_line[:, 1], self.ref_line[:, 2], self.ref_yaw, self.ref_s, self.ref_kappa