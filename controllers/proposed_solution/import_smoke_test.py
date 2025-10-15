# import_smoke_test.py
# 作用：只做“导入自检”，不跑仿真、不画图。
import os, sys, time, importlib

BASE = os.path.dirname(__file__)
PYROBOTICS_ROOT = os.path.join(BASE, "third_party", "PythonRobotics")
sys.path.append(PYROBOTICS_ROOT)

def check(modname: str, obj: str | None = None) -> None:
    t0 = time.time()
    try:
        mod = importlib.import_module(modname)
        if obj:
            getattr(mod, obj)
        dt = (time.time() - t0) * 1000.0
        print(f"OK  | import {modname}{'.'+obj if obj else ''}  ({dt:.1f} ms)")
    except Exception as e:
        print(f"FAIL| import {modname}{'.'+obj if obj else ''} -> {type(e).__name__}: {e}")

def main():
    print(f"[Python] {sys.executable}")
    print(f"[sys.path+] {PYROBOTICS_ROOT}")
    print("-" * 72)

    checks = [
        # 映射 / 建图
        ("Mapping.ray_casting_grid_map.ray_casting_grid_map", None),

        # SLAM
        ("SLAM.EKFSLAM.ekf_slam", None),             # 需要 scipy
        ("SLAM.FastSLAM2.fast_slam2", None),         # 注意 fast_slam2.py（有下划线）

        # 定位（MCL/直方图滤波）
        ("Localization.particle_filter.particle_filter", None),
        ("Localization.histogram_filter.histogram_filter", None),

        # 路径规划
        ("PathPlanning.AStar.a_star", None),
        ("PathPlanning.Dijkstra.dijkstra", None),

        # 轨迹跟踪（你的仓库是 stanley_control 版）
        ("PathTracking.stanley_control.stanley_control", None),

        # 工具
        ("utils.angle", None),                       # 需要 scipy
    ]

    for mod, obj in checks:
        check(mod, obj)

if __name__ == "__main__":
    main()
