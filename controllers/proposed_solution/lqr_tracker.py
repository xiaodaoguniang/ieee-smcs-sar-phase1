# controllers/proposed_solution/lqr_tracker.py
# -*- coding: utf-8 -*-
"""
LQR-like differential-drive path tracking (v, w) for Webots robots.

核心思路（与你现有控制器保持一致）：
- 选取“前视点”（lookahead 点），计算指向角 gth；
- 角误差 dth = wrap(gth - yaw)，角速度 w = Kp * dth （限幅）；
- 线速度 v = clip( Vcap * max(0.2, cos(dth)) )，在目标附近动态降速；
- 动态前视距离：随目标距离缩放（近目标→更短的前视，提高收敛稳定性）。

提供类：
- LQRTracker：带参数、动态前视、饱和与若干辅助判断（到点、路径为空）。

使用：
    tracker = LQRTracker(max_v=0.7, max_w=1.8, k_th=1.2,
                         lookahead=0.35, v_slow=0.6)
    v_cmd, w_cmd, info = tracker.compute_cmd(
        x, y, yaw,
        path_xy,     # [(x,y), ...]
        goal_xy=(gx, gy),
        dt=0.032     # 可选；仅用于内部滤波/保留，当前版本不用也行
    )
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple, Dict, Optional

Vec2 = Tuple[float, float]

def wrap_angle(a: float) -> float:
    """wrap to (-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi

def pick_lookahead(px: float, py: float, path_xy: List[Vec2], lookahead: float) -> Vec2:
    """
    在 path 上寻找距离当前位置约为 lookahead 的目标点。
    实现与你之前的一致：先找最近路径点，再沿路径累计弧长到阈值位置。
    """
    if not path_xy:
        return (px, py)
    # 最近点下标
    best_i, best_d2 = 0, float("inf")
    for i, (x, y) in enumerate(path_xy):
        d2 = (x - px) * (x - px) + (y - py) * (y - py)
        if d2 < best_d2:
            best_d2, best_i = d2, i
    # 从最近点往后累积
    acc = 0.0
    j = best_i
    for k in range(best_i, len(path_xy) - 1):
        dx = path_xy[k + 1][0] - path_xy[k][0]
        dy = path_xy[k + 1][1] - path_xy[k][1]
        seg = math.hypot(dx, dy)
        acc += seg
        j = k + 1
        if acc >= max(lookahead, 1e-6):
            break
    return path_xy[j]

@dataclass
class LQRParams:
    # 限幅
    max_v: float = 0.7
    max_w: float = 1.8
    # LQR（比例）与速度调节
    k_th: float = 1.2            # 角误差比例系数 → w
    v_slow: float = 0.6          # 线速度基础系数（可再乘 cos(dth) 衰减）
    # 前视
    lookahead: float = 0.35      # 基础前视（m）
    min_look: float = 0.18       # 下限
    # 近目标降速
    goal_slow_radius: float = 0.8   # 0~此距离做线速封顶渐变
    v_min_cos: float = 0.2          # cos(dth) 衰减下的最低系数（防止完全停转）
    # 到点判定（可用于外层停止/切目标）
    goal_tol: float = 0.10
    tail_tol: float = 0.35

class LQRTracker:
    def __init__(self, **kwargs):
        self.p = LQRParams(**kwargs)

    def compute_cmd(
        self,
        x: float, y: float, yaw: float,
        path_xy: List[Vec2],
        goal_xy: Optional[Vec2] = None,
        dt: Optional[float] = None
    ) -> Tuple[float, float, Dict]:
        """
        返回 (v_cmd, w_cmd, info)
        - 若 path 为空，返回 (0, 0)
        - info 包含用于调试/可视化的内部量
        """
        info: Dict = {"ok": False, "reason": ""}

        if not path_xy:
            info["reason"] = "empty_path"
            return 0.0, 0.0, info

        gx, gy = goal_xy if goal_xy else path_xy[-1]
        dist_goal = math.hypot(gx - x, gy - y)
        # 路径尾点距离
        tx_last, ty_last = path_xy[-1]
        dist_tail = math.hypot(tx_last - x, ty_last - y)

        # 到点判定，仅给出信息；停止逻辑由上层决定
        info["arrive_goal"] = bool(dist_goal < self.p.goal_tol)
        info["arrive_tail"] = bool(dist_tail < self.p.tail_tol)

        # 动态前视：近目标减小，远目标增大
        near = max(0.0, min(1.0, dist_goal / max(1e-6, self.p.goal_slow_radius)))  # 0:近 → 1:远
        dyn_look = max(self.p.min_look, self.p.lookahead * (0.5 + 0.5 * near))

        # 前视点
        tx, ty = pick_lookahead(x, y, path_xy, dyn_look)

        # 指向角与角误差
        gth = math.atan2(ty - y, tx - x)
        dth = wrap_angle(gth - yaw)

        # 线速封顶：近目标更小
        v_cap = min(self.p.max_v, 0.25 + 0.45 * near)
        # 基础线速（含 cos 衰减，避免大转弯时冲过头）
        v_cmd = min(v_cap, self.p.v_slow * max(self.p.v_min_cos, math.cos(dth)))
        # 角速度（比例项）
        w_cmd = max(-self.p.max_w, min(self.p.max_w, self.p.k_th * dth))

        info.update({
            "ok": True,
            "dist_goal": dist_goal,
            "dist_tail": dist_tail,
            "dyn_look": dyn_look,
            "target_xy": (tx, ty),
            "gth": gth,
            "dth": dth,
            "v_cap": v_cap,
        })
        return v_cmd, w_cmd, info
