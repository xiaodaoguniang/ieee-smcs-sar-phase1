# -*- coding: utf-8 -*-
"""
lidar_gridmap.py
把一帧/多帧 LiDAR 扫描融合到 2D 栅格 (log-odds) 的小工具。

坐标&角度约定（与 Webots/ROS 下常见约定一致）：
- 平面坐标用 (x, y)；yaw 为 z 轴朝上时的偏航角，逆时针为正。
- Webots 中 RPLidar-类激光的“角数组”通常是按顺时针增长的；因此
  组合时请用 th = yaw - a。若你的激光角度是 CCW，请把 ccw=True。
- 绘图：imshow(origin='lower', extent=[XMIN,XMAX,YMIN,YMAX]) → y 轴向上。

模块功能：
- GridMap：维护 log-odds 栅格、融合一帧/多帧扫描、导出 P(occ)/PNG。
- 工具函数：make_angles_webots() 生成与 Webots Lidar 一致的角数组。
"""
from __future__ import annotations
import math, os
import numpy as np

# ----------------- 小工具 -----------------
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def bresenham(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    out = []
    while True:
        out.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc
    return out

def logistic_safe(L: np.ndarray) -> np.ndarray:
    out = np.empty_like(L, dtype=np.float32)
    pos = L >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-L[pos]))
    neg = ~pos
    e = np.exp(L[neg]); out[neg] = e / (1.0 + e)
    return out

def make_angles_webots(fov: float, n_ray: int) -> np.ndarray:
    """
    生成和 Webots Lidar.getRangeImage() 对齐的角数组：
    - 最左射线角为 -FOV/2，最右为 +FOV/2；
    - Webots 的图像索引从左到右，几何上是顺时针（CW）；
    - 调用 GridMap.integrate_scan(..., ccw=False) 即 th = yaw - a。
    """
    if n_ray <= 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-fov/2.0, +fov/2.0, n_ray, dtype=float)

# ----------------- 主类 -----------------
class GridMap:
    """
    维护世界坐标系下的 log-odds 栅格。
    """
    def __init__(self,
                 xmin: float, xmax: float,
                 ymin: float, ymax: float,
                 res: float,
                 l_free: float = -0.35,
                 l_occ: float  = +1.10,
                 l_min: float  = -8.0,
                 l_max: float  = +8.0):
        assert xmax > xmin and ymax > ymin and res > 0
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.res = float(res)
        self.l_free, self.l_occ = float(l_free), float(l_occ)
        self.l_min,  self.l_max = float(l_min),  float(l_max)

        self.H = int(math.ceil((self.ymax - self.ymin) / self.res))
        self.W = int(math.ceil((self.xmax - self.xmin) / self.res))
        self.L = np.zeros((self.H, self.W), dtype=np.float32)

    # ---- 坐标换算 ----
    def world_to_rc(self, wx: float, wy: float) -> tuple[int, int]:
        c = int((wx - self.xmin) / self.res)
        r = int((wy - self.ymin) / self.res)
        return r, c

    def in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < self.H) and (0 <= c < self.W)

    # ---- 融合一帧扫描 ----
    def integrate_scan(self,
                       x: float, y: float, yaw: float,
                       angles: np.ndarray, ranges: np.ndarray,
                       rmin: float, rmax: float,
                       *,
                       ccw: bool = False,
                       end_occ_margin: float = 0.20):
        """
        把一帧 LiDAR 融合到 log-odds 网格。
        参数：
          - (x,y,yaw): 机器人位姿（世界系，yaw 逆时针为正）。
          - angles: 每条射线相对车体前向的角，长度 = ranges。
          - ranges: 距离数组，单位 m。
          - rmin/rmax: 传感器有效量程。
          - ccw: 若 True，表示角数组为“逆时针增大”（常见于ROS激光）；此时用 th = yaw + a。
                 若 False（Webots 常见），用 th = yaw - a。
          - end_occ_margin: 末端命中 “占据” 时，从 rmin 略退以避免近场假阳性。
        """
        if ranges is None or angles is None:
            return
        assert len(angles) == len(ranges)

        r0, c0 = self.world_to_rc(x, y)
        if not self.in_bounds(r0, c0):
            return

        for a, r in zip(angles, ranges):
            if r is None or (not math.isfinite(r)):
                continue
            if r <= max(0.05, float(rmin)) or r > float(rmax):
                continue

            # 关键：组合全局角
            th = yaw + a if ccw else yaw - a

            wx_end = x + r * math.cos(th)
            wy_end = y + r * math.sin(th)
            r1, c1 = self.world_to_rc(wx_end, wy_end)

            ray = bresenham(r0, c0, r1, c1)
            if len(ray) >= 2:
                # 中间 free
                for rr, cc in ray[1:-1]:
                    if self.in_bounds(rr, cc):
                        self.L[rr, cc] = float(np.clip(self.L[rr, cc] + self.l_free,
                                                       self.l_min, self.l_max))
                # 末端占据（略过极近距离）
                if r > (float(rmin) + end_occ_margin):
                    rr, cc = ray[-1]
                    if self.in_bounds(rr, cc):
                        self.L[rr, cc] = float(np.clip(self.L[rr, cc] + self.l_occ,
                                                       self.l_min, self.l_max))

    # ---- 融合多帧（批处理）----
    def integrate_batch(self, pose_scan_list, *, ccw: bool = False,
                        rmin: float = 0.1, rmax: float = 12.0):
        """
        pose_scan_list: 可迭代 [(x,y,yaw, angles, ranges), ...]
        """
        for (x, y, yaw, angles, ranges) in pose_scan_list:
            self.integrate_scan(x, y, yaw, angles, ranges, rmin, rmax, ccw=ccw)

    # ---- 导出与预览 ----
    def probs(self) -> np.ndarray:
        """返回 P(occ) 概率栅格（float32, 0..1）。"""
        return logistic_safe(self.L)

    def save_png(self, path: str, title: str | None = None):
        """保存预览 PNG：origin='lower', extent=世界坐标范围。"""
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        P = self.probs()
        plt.figure(figsize=(6, 4.5), dpi=150)
        plt.imshow(P, origin='lower',
                   extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                   vmin=0.0, vmax=1.0, cmap='gray')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        if title: plt.title(title)
        cbar = plt.colorbar(); cbar.set_label('P(occ)')
        plt.tight_layout(); plt.savefig(path); plt.close()

    def save_numpy(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.L)

    # ---- 方便组合多机器人（以“最大”合并）----
    def fuse_max(self, other_L: np.ndarray):
        """把另一张 log-odds 栅格融合进来（逐格取最大值）。"""
        assert other_L.shape == self.L.shape
        np.maximum(self.L, other_L, out=self.L)
