import numpy as np
def spherical_to_cartesian(r, theta, phi):
    """将球坐标转换为笛卡尔坐标"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])
def ang_calc(theta1, phi1, theta2, phi2):
        theta1 = theta1 / 180 * np.pi
        phi1 = phi1 / 180 * np.pi
        theta2 = theta2 / 180 * np.pi
        phi2 = phi2 / 180 * np.pi
        # 计算夹角的余弦值
        if theta1 == theta2:
            # 如果极角相同，直接计算方位角的差值
            angle = abs(phi1 - phi2)
            # 确保夹角在0到π之间
            if angle > np.pi:
                angle = 2 * np.pi - angle
            return angle
        else:
            r1=r2=1
            # 将球坐标转换为笛卡尔坐标
            A = spherical_to_cartesian(r1, theta1, phi1)
            B = spherical_to_cartesian(r2, theta2, phi2)

            # 计算点积
            dot_product = np.dot(A, B)

            # 计算模
            magnitude_A = np.linalg.norm(A)
            magnitude_B = np.linalg.norm(B)

            # 计算夹角
            cos_theta = dot_product / (magnitude_A * magnitude_B)

            # 处理可能的浮点数误差
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angle = np.arccos(cos_theta)  # 结果以弧度表示

            return angle