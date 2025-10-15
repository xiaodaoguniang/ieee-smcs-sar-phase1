# controllers/proposed_solution/proposed_solution.py 
# -*- coding: utf-8 -*-

from controller import Robot
import os, time, json, math, itertools
import numpy as np

from a_star_adapted import AStarGrid                       # A*
from lidar_gridmap import GridMap, make_angles_webots      # LiDAR -> log-odds
from lqr_tracker import LQRTracker                         # LQR path tracking (v,w)
from victim_detector_xy import spin_scan_and_log, cluster_xy_points  # Visual Detection + Clustering

# ===== Map Area and Resolution =====
XMIN, XMAX = -6.5, 6.9
YMIN, YMAX = -3.1, 5.0
RES        = 0.045   # 4.5 cm / cell

# ===== log-odds parameters =====
L_FREE, L_OCC = -0.35, +1.10
L_MIN,  L_MAX = -8.0,  +8.0

# ===== Binarization =====
OCC_THRESH    = 0.65   

# ===== LiDAR In-Place Rotation Sampling (Mapping)=====
N_SNAP      = 8
WAIT_S      = 0.20
PREWARM     = 5
SPIN_W_CMD  = 0.5

# ===== Online Replanning Trigger =====
REPLAN_DIST_M        = 0.38     
REPLAN_SPINS         = 3       
FRONT_BLOCK_AHEAD_M  = 0.6

# ===== Robot Parameters / Safety =====
ROBOT_RADIUS = 0.17
EXTRA_MARGIN = 0.08     # ★ 改为 0.08
GOAL_TOL     = 0.13     # 保持与之前一致的“判定到达”阈值（你上一问已确认）

# ===== 多机 & 合并参数 =====
ROBOT_NAMES_CANON = ["ROSbot 1", "ROSbot 2", "ROSbot 3"]
SQUAD_WAIT_S = 1.5
MERGE_EPS    = 0.60
MAX_VICTIMS  = 3

# ===== 底盘（差速）=====
WHEEL_RADIUS = 0.033
WHEEL_BASE   = 0.20
MAX_V        = 0.5    # ★ 改为 0.5
MAX_W        = 2.0    # ★ 改为 2.0

# ---------- 基础工具 ----------
def logistic_safe(L: np.ndarray) -> np.ndarray:
    out = np.empty_like(L, dtype=np.float32)
    pos = L >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-L[pos]))
    neg = ~pos
    e = np.exp(L[neg]); out[neg] = e / (1.0 + e)
    return out

def fmt_xy(x: float, y: float) -> str:
    return f"({x:+.2f},{y:+.2f})"

def inflate_obstacles(occ: np.ndarray, k: int) -> np.ndarray:
    """棋盘距离膨胀 k 格（方形结构元素）"""
    if k <= 0: return occ
    H, W = occ.shape
    out = occ.copy()
    for dr in range(-k, k+1):
        for dc in range(-k, k+1):
            if dr == 0 and dc == 0: continue
            r_src0 = max(0, -dr); r_src1 = min(H, H - dr)
            c_src0 = max(0, -dc); c_src1 = min(W, W - dc)
            r_dst0 = max(0,  dr); r_dst1 = min(H, H + dr)
            c_dst0 = max(0,  dc); c_dst1 = min(W, W + dc)
            if r_src1 > r_src0 and c_src1 > c_src0:
                out[r_dst0:r_dst1, c_dst0:c_dst1] = np.maximum(
                    out[r_dst0:r_dst1, c_dst0:c_dst1],
                    occ[r_src0:r_src1, c_src0:c_src1]
                )
    return out

def gm_world_to_rc(gm: GridMap, x: float, y: float) -> tuple[int, int]:
    c = int((x - gm.xmin) / gm.res)
    r = int((y - gm.ymin) / gm.res)
    return r, c

# ---------- 通信（机器人-机器人） ----------
def get_comms(robot, timestep):
    try:
        rx = robot.getDevice("robot to robot receiver"); rx.enable(timestep)
    except Exception:
        rx = None
    try:
        tx = robot.getDevice("robot to robot emitter")
    except Exception:
        tx = None
    return rx, tx

def r2r_send(tx, payload: dict):
    if not tx: return
    try:
        tx.send(json.dumps(payload).encode())
    except Exception:
        pass

def poll_r2r(robot, rx):
    """非阻塞清空 r2r 收件箱，返回列表"""
    out = []
    if not rx: return out
    try:
        while rx.getQueueLength() > 0:
            s = rx.getString(); rx.nextPacket()
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return out

def broadcast_victims(tx, robot_name, my_pos, victims_xy):
    r2r_send(tx, {
        "type": "victim_centers",
        "robot_id": robot_name,
        "timestamp": time.time(),
        "robot_pos": list(map(float, my_pos)),
        "victims_xy": [[float(x), float(y)] for (x, y) in victims_xy],
    })

# ---------- 通信（机器人-监督端，模仿 simple_rosbot_sar.py） ----------
def get_supervisor_comms(robot, timestep):
    try:
        rx = robot.getDevice("supervisor receiver"); rx.enable(timestep)
    except Exception:
        rx = None
    try:
        tx = robot.getDevice("supervisor emitter")
    except Exception:
        tx = None
    return rx, tx

def notify_supervisor_victim_reached(robot, sup_tx, robot_name, pos_xyz, victim_xy):
    if not sup_tx: return
    payload = {
        "timestamp": robot.getTime(),
        "robot_id": robot_name,
        "position": list(map(float, pos_xyz)),     # [x, y, z]
        "intended_action": "victim_reached",
        "reason": f"Reached assigned victim at ({victim_xy[0]:.2f},{victim_xy[1]:.2f})",
        "victim_found": True,
        "victim_confidence": 1.0
    }
    try:
        sup_tx.send(json.dumps(payload).encode())
    except Exception:
        pass

# ---------- 合并 & 分配 ----------
def merge_points_eps(points, eps=MERGE_EPS, max_clusters=MAX_VICTIMS):
    clusters = []
    for (x, y) in points:
        matched = False
        for c in clusters:
            cx = c["sum"][0] / c["n"]; cy = c["sum"][1] / c["n"]
            if math.hypot(x - cx, y - cy) <= eps:
                c["sum"][0] += x; c["sum"][1] += y; c["n"] += 1
                matched = True; break
        if not matched:
            clusters.append({"sum":[x,y], "n":1})
    centers = [(c["sum"][0]/c["n"], c["sum"][1]/c["n"]) for c in clusters]
    centers = sorted(centers, key=lambda p: (p[0], p[1]))[:max_clusters]
    return centers

def unique_nearest_assignment(robots_ordered, robot_pos_by_name, victims_xy):
    R = [r for r in robots_ordered if r in robot_pos_by_name]
    m = min(len(R), len(victims_xy))
    assigns = {}
    if m == 0: return assigns
    Rm = R[:m]; Vm = victims_xy[:m]
    best_cost, best_perm = float("inf"), None
    for perm in itertools.permutations(range(m)):
        cost = 0.0
        for i, pi in enumerate(perm):
            rx, ry = robot_pos_by_name[Rm[i]]
            vx, vy = Vm[pi]
            cost += math.hypot(vx - rx, vy - ry)
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    for i, pi in enumerate(best_perm):
        assigns[Rm[i]] = Vm[pi]
    return assigns

# ---------- Matplotlib 出图 ----------
def save_png_matrix(mat, extent, path, title, cmap='gray', vmin=None, vmax=None,
                    scatter=None, lines=None, annots=None, legend=True):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4.5), dpi=150)
        plt.imshow(mat, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        handles = []
        if scatter:
            for xs, ys, kw in scatter:
                if xs and ys:
                    h = plt.scatter(xs, ys, **kw)
                    if "label" in kw and kw["label"]:
                        handles.append(h)
        if lines:
            for xs, ys, kw in (lines or []):
                if xs and ys:
                    plt.plot(xs, ys, **kw)
        if annots:
            for (x, y, text, kw) in annots:
                plt.annotate(text, (x, y), **kw)
        if legend and handles:
            plt.legend(loc="upper right", framealpha=0.75)
        plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title(title)
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[SAVE ] {title} → {path}")
    except Exception as e:
        print(f"[WARN ] save png failed: {e}")

# ====== 车轮控制 ======
def try_get_wheels(robot):
    cands = {
        "fl": ["fl_wheel_joint", "front left wheel motor", "left_front_wheel"],
        "fr": ["fr_wheel_joint", "front right wheel motor", "right_front_wheel"],
        "rl": ["rl_wheel_joint", "rear left wheel motor", "left_rear_wheel"],
        "rr": ["rr_wheel_joint", "rear right wheel motor", "right_rear_wheel"],
    }
    motors = {}
    for k, names in cands.items():
        m = None
        for nm in names:
            try:
                m = robot.getDevice(nm)
                if m:
                    m.setPosition(float('inf'))
                    m.setVelocity(0.0)
                    break
            except Exception:
                m = None
        motors[k] = m
    left  = [m for m in (motors["fl"], motors["rl"]) if m]
    right = [m for m in (motors["fr"], motors["rr"]) if m]
    return left, right

def send_vw(left_group, right_group, v, w):
    v = max(-MAX_V, min(MAX_V, float(v)))
    w = max(-MAX_W, min(MAX_W, float(w)))
    v_l = v - 0.5 * w * WHEEL_BASE
    v_r = v + 0.5 * w * WHEEL_BASE
    wl  = v_l / WHEEL_RADIUS
    wr  = v_r / WHEEL_RADIUS
    for m in left_group:  m.setVelocity(wl)
    for m in right_group: m.setVelocity(wr)

def spin_in_place(robot, left_group, right_group, w_cmd: float, step_ms: int, dur_s: float):
    if not left_group or not right_group or w_cmd == 0.0:
        t_end = robot.getTime() + dur_s
        while robot.step(step_ms) != -1 and robot.getTime() < t_end:
            pass
        return
    v_l = -0.5 * w_cmd * WHEEL_BASE
    v_r = +0.5 * w_cmd * WHEEL_BASE
    wl  = v_l / WHEEL_RADIUS
    wr  = v_r / WHEEL_RADIUS
    for m in left_group:  m.setVelocity(wl)
    for m in right_group: m.setVelocity(wr)
    t_end = robot.getTime() + dur_s
    while robot.step(step_ms) != -1 and robot.getTime() < t_end:
        pass
    for m in left_group + right_group:
        m.setVelocity(0.0)

# ---------- 起终点吸附（到最近 free cell） ----------
def snap_xy_to_free(gm: GridMap, occ: np.ndarray, x: float, y: float,
                    max_radius_m: float = 1.0) -> tuple[float, float]:
    H, W = occ.shape
    r, c = gm_world_to_rc(gm, x, y)
    if 0 <= r < H and 0 <= c < W and occ[r, c] == 0:
        return x, y
    max_k = max(1, int(max_radius_m / gm.res))
    for k in range(1, max_k + 1):
        rmin, rmax = max(0, r - k), min(H - 1, r + k)
        cmin, cmax = max(0, c - k), min(W - 1, c + k)
        for cc in range(cmin, cmax + 1):
            for rr in (rmin, rmax):
                if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] == 0:
                    wx = gm.xmin + (cc + 0.5) * gm.res
                    wy = gm.ymin + (rr + 0.5) * gm.res
                    return wx, wy
        for rr in range(rmin + 1, rmax):
            for cc in (cmin, cmax):
                if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] == 0:
                    wx = gm.xmin + (cc + 0.5) * gm.res
                    wy = gm.ymin + (rr + 0.5) * gm.res
                    return wx, wy
    return x, y

# ---------- 小余量 clearance 检查（可关） ----------
def path_has_clearance(gm: GridMap, occ: np.ndarray, path_xy: list[tuple[float, float]],
                       radius_m: float) -> bool:
    if (not path_xy) or radius_m <= 0.0: return True
    k = max(1, int(math.ceil(radius_m / gm.res)))
    H, W = occ.shape
    rad2 = k * k
    for (x, y) in path_xy:
        r, c = gm_world_to_rc(gm, x, y)
        if not (0 <= r < H and 0 <= c < W): return False
        if occ[r, c] == 1: return False
        r0, r1 = max(0, r - k), min(H - 1, r + k)
        c0, c1 = max(0, c - k), min(W - 1, c + k)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                dr = rr - r; dc = cc - c
                if (dr*dr + dc*dc) <= rad2 and occ[rr, cc] == 1:
                    return False
    return True

# ---------- 前方阻塞（在 occ_plan 上看未来 FRONT_BLOCK_AHEAD_M 是否撞格） ----------
def path_front_blocked(gm: GridMap, occ_plan: np.ndarray, path_xy, x, y, ahead_m=FRONT_BLOCK_AHEAD_M):
    if not path_xy: return True
    best_i, best_d = 0, 1e18
    for i,(px,py) in enumerate(path_xy):
        d = (px-x)*(px-x)+(py-y)*(py-y)
        if d < best_d: best_d, best_i = d, i
    acc = 0.0
    for k in range(best_i, len(path_xy)-1):
        ax, ay = path_xy[k]
        bx, by = path_xy[k+1]
        seg = math.hypot(bx-ax, by-ay)
        steps = max(1, int(seg / gm.res))
        for s in range(steps+1):
            tx = ax + (bx-ax)*(s/steps)
            ty = ay + (by-ay)*(s/steps)
            r, c = gm_world_to_rc(gm, tx, ty)
            if 0 <= r < occ_plan.shape[0] and 0 <= c < occ_plan.shape[1]:
                if occ_plan[r, c] == 1:
                    return True
        acc += seg
        if acc >= ahead_m: break
    return False

# ---------- 主流程 ----------
def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep()) or 32
    name = robot.getName()

    # 设备
    gps = robot.getDevice("gps"); gps.enable(step)
    lidar = robot.getDevice("laser"); lidar.enable(step)

    # IMU 优先 inertial_unit，否则回退 compass
    imu = None
    try:
        imu = robot.getDevice("imu inertial_unit")
        imu.enable(step)
    except Exception:
        imu = None
    compass = None
    if imu is None:
        try:
            compass = robot.getDevice("imu compass")
            compass.enable(step)
        except Exception:
            compass = None

    def get_yaw():
        if imu:
            try:
                return imu.getRollPitchYaw()[2]
            except Exception:
                pass
        if compass:
            try:
                n = compass.getValues()
                return math.atan2(n[0], n[1])
            except Exception:
                pass
        return 0.0

    # LiDAR 角度（Webots 顺时针）→ integrate_scan(ccw=False）
    Nray = lidar.getHorizontalResolution()
    FOV  = lidar.getFov()
    RMAX = lidar.getMaxRange()
    RMIN = lidar.getMinRange()
    angles = make_angles_webots(FOV, Nray)

    # 通信（r2r + supervisor）
    rx, tx = get_comms(robot, step)
    sup_rx, sup_tx = get_supervisor_comms(robot, step)

    print(f"[LIDAR] FOV={math.degrees(FOV):.1f}° Rays={Nray}  Range=[{RMIN:.2f},{RMAX:.2f}] m")
    print(f"[MAP  ] bounds x:[{XMIN},{XMAX}] y:[{YMIN},{YMAX}]  res={RES} m/cell")
    print(f"[ROBOT] radius = {ROBOT_RADIUS:.3f} m")
    outdir = os.path.join(os.getcwd(), "map_out"); os.makedirs(outdir, exist_ok=True)
    ts = int(time.time())

    # ========== ① 视觉：原地自转检测 & 本机聚类 ==========
    print(f"[{name}] Step-1: vision spin & detect...")
    _summary = spin_scan_and_log(robot,
                                 STEP_DEG=45.0, TOTAL_DEG=360.0,
                                 WHEEL_SPEED=3.0, SAFE_TIMEOUT=60.0)

    clusters, _ = cluster_xy_points(
        jsonl_glob=f"shots_{name}/*victim_detections_*.jsonl",
        depth_ok=(0.60, 8.0),
        min_mask_pixels=2000,
        min_h_frac=0.50,
        cluster_eps=0.60,
        cluster_min_pts=2
    )
    my_victims_xy = [tuple(c["center"]) for c in sorted(clusters, key=lambda c: (c["center"][0], c["center"][1]))][:MAX_VICTIMS]
    print(f"[{name}] Local clusters: {[fmt_xy(x,y) for (x,y) in my_victims_xy]}")

    # ========== ② 广播本机结果 + 接收队友 ==========
    sx, sy, _ = gps.getValues()
    broadcast_victims(tx, name, (sx,sy), my_victims_xy)
    # 简单收一次大家的侦察结果
    t_wait = robot.getTime() + SQUAD_WAIT_S
    squad_msgs = []
    while robot.getTime() < t_wait and robot.step(step) != -1:
        squad_msgs.extend(poll_r2r(robot, rx))

    robot_pos_by_name = {name: (sx, sy)}
    all_points = list(my_victims_xy)
    for m in squad_msgs:
        if isinstance(m, dict) and m.get("type") == "victim_centers":
            rid = m.get("robot_id"); rpos = m.get("robot_pos", [None,None,None])
            vxs = m.get("victims_xy", [])
            if isinstance(rid, str) and isinstance(vxs, list):
                if isinstance(rpos, list) and len(rpos) >= 2 and all(isinstance(v, (int,float)) for v in rpos[:2]):
                    robot_pos_by_name[rid] = (float(rpos[0]), float(rpos[1]))
                for it in vxs:
                    if isinstance(it, list) and len(it) >= 2:
                        all_points.append((float(it[0]), float(it[1])))

    robots_sorted = [r for r in sorted(ROBOT_NAMES_CANON) if r in robot_pos_by_name]
    global_victims = merge_points_eps(all_points, eps=MERGE_EPS, max_clusters=MAX_VICTIMS)
    print(f"[{name}] Global victims (merged): {[fmt_xy(x,y) for (x,y) in global_victims]}")

    # 给 victim 编稳定 id（1..N）；构建 victim->robot 初始分配表
    vid2xy = {i+1: (x,y) for i,(x,y) in enumerate(global_victims)}
    assignments = unique_nearest_assignment(robots_sorted, robot_pos_by_name, global_victims)

    # invert: 通过“最近”匹配 victim id（避免浮点相等）
    def nearest_vid(xy):
        best, bid = 1e18, None
        for vid,(vx,vy) in vid2xy.items():
            d = (xy[0]-vx)**2 + (xy[1]-vy)**2
            if d < best: best, bid = d, vid
        return bid

    victim_owner = {}  # vid -> robot_name
    for rname, vxy in assignments.items():
        vid = nearest_vid(vxy)
        if vid is not None: victim_owner[vid] = rname

    my_assign = assignments.get(name, None)
    if not my_assign:
        print(f"[ASSIGN] {name}: no target, exit.")
        return
    my_vid = nearest_vid(my_assign)
    print(f"[ASSIGN] {name} -> VID#{my_vid} {fmt_xy(*my_assign)} (unique-nearest)")

    # 完成集（所有人共享 via 广播）
    done_vids = set()

    # ========== ③ 建图（抽 N 帧）==========
    gm = GridMap(XMIN, XMAX, YMIN, YMAX, RES, l_free=L_FREE, l_occ=L_OCC, l_min=L_MIN, l_max=L_MAX)
    for _ in range(PREWARM):
        if robot.step(step) == -1: return

    left_group, right_group = try_get_wheels(robot)
    for _ in range(N_SNAP):
        spin_in_place(robot, left_group, right_group, SPIN_W_CMD, step, WAIT_S)
        rx_, ry_, _ = gps.getValues()
        yaw = get_yaw()
        ranges = np.asarray(lidar.getRangeImage(), dtype=float)
        gm.integrate_scan(rx_, ry_, yaw, angles, ranges, RMIN, RMAX, ccw=False)

    # 概率 & 二值
    sx, sy, _ = gps.getValues()
    extent = [gm.xmin, gm.xmax, gm.ymin, gm.ymax]
    P = logistic_safe(gm.L)
    occ_raw = (P > OCC_THRESH).astype(np.uint8)

    # ========== ④ 初次 A* ==========
    PLAN_MARGIN = 0.05
    k_plan = int(math.ceil((ROBOT_RADIUS + PLAN_MARGIN) / RES)) + 2
    k_exec = k_plan + 1

    print(f"[PLAN ] inflation k_plan={k_plan} cells (radius={ROBOT_RADIUS:.3f} m, res={RES:.3f} m)")
    occ_plan = inflate_obstacles(occ_raw, k_plan)

    gx_raw, gy_raw = vid2xy[my_vid]
    sx_fix, sy_fix = snap_xy_to_free(gm, occ_plan, sx, sy, max_radius_m=1.0)
    gx_fix, gy_fix = snap_xy_to_free(gm, occ_plan, gx_raw, gy_raw, max_radius_m=1.0)
    planner = AStarGrid(XMIN, XMAX, YMIN, YMAX, RES)
    path_xy = planner.plan(occ_plan, (sx_fix, sy_fix), (gx_fix, gy_fix))
    if not path_xy or len(path_xy) < 2:
        print(f"[PLAN ] initial A* FAIL, exit.")
        return
    if not path_has_clearance(gm, occ_raw, path_xy, EXTRA_MARGIN):
        print(f"[CHECK] extra-margin ({EXTRA_MARGIN:.2f} m) clearance FAIL at init -> exit.")
        return

    tracker = LQRTracker(max_v=MAX_V, max_w=MAX_W)
    acc_since_replan = 0.0
    last_x, last_y = sx, sy

    # === 协商消息名 ===
    # victim_done: {type, vid, robot_id}
    # handover_request: {type, vid, proposer, proposer_dist}
    # handover_ack: {type, vid, from, accepted}

    def try_pick_next_vid(cur_x, cur_y):
        cand = None; best = 1e18
        for vid,(vx,vy) in vid2xy.items():
            if vid in done_vids: continue
            if vid == my_vid: continue
            d2 = (vx-cur_x)**2 + (vy-cur_y)**2
            if d2 < best:
                best, cand = d2, vid
        return cand

    # 轻量重建 & 规划（复用你的流程）
    def lightweight_refresh_and_replan(cur_x, cur_y, goal_xy):
        for _ in range(REPLAN_SPINS):
            spin_in_place(robot, left_group, right_group, w_cmd=0.5, step_ms=step, dur_s=WAIT_S)
            rx_, ry_, _ = gps.getValues()
            _yaw = get_yaw()
            ranges = np.asarray(lidar.getRangeImage(), dtype=float)
            gm.integrate_scan(rx_, ry_, _yaw, angles, ranges, RMIN, RMAX, ccw=False)
        P_ = logistic_safe(gm.L)
        occ_raw_ = (P_ > OCC_THRESH).astype(np.uint8)
        occ_plan_ = inflate_obstacles(occ_raw_, k_plan)
        sx2, sy2 = snap_xy_to_free(gm, occ_plan_, cur_x, cur_y, max_radius_m=1.0)
        gx2, gy2 = snap_xy_to_free(gm, occ_plan_, goal_xy[0], goal_xy[1], max_radius_m=1.0)
        new_path = planner.plan(occ_plan_, (sx2, sy2), (gx2, gy2))
        if new_path and len(new_path) >= 2 and path_has_clearance(gm, occ_raw_, new_path, EXTRA_MARGIN):
            return new_path, occ_raw_, occ_plan_, (gx2, gy2)
        return None, occ_raw_, occ_plan_, (goal_xy[0], goal_xy[1])

    # ====== 主循环 ======
    while robot.step(step) != -1:
        # 处理 r2r 收件箱（victim_done / handover_request / handover_ack）
        for m in poll_r2r(robot, rx):
            if not isinstance(m, dict): continue
            t = m.get("type","")
            if t == "victim_done":
                did = int(m.get("vid", -1))
                if did > 0: done_vids.add(did)
                # 若我们正跟踪此受害者，直接停止并挑新目标
                if did == my_vid:
                    print(f"[INFO ] VID#{did} done by {m.get('robot_id')} → switching target")
                    send_vw(left_group, right_group, 0.0, 0.0)
                    nxt = try_pick_next_vid(*gps.getValues()[:2])
                    if nxt is None:
                        print("[INFO ] no remaining victim, exit.")
                        return
                    my_vid = nxt
                    gx_raw, gy_raw = vid2xy[my_vid]
                    path_xy, occ_raw, occ_plan, (gx_fix, gy_fix) = lightweight_refresh_and_replan(
                        *gps.getValues()[:2], (gx_raw, gy_raw)
                    )
                    if not path_xy:
                        print("[PLAN ] replan FAIL after done-switch, exit.")
                        return
                    print(f"[SWITCH] new VID#{my_vid} {fmt_xy(gx_raw, gy_raw)}")
                    acc_since_replan = 0.0

            elif t == "handover_request":
                req_vid = int(m.get("vid", -1))
                proposer = m.get("proposer","")
                proposerd = float(m.get("proposer_dist", 1e9))
                if req_vid == my_vid and proposer and proposer != name:
                    x,y,_ = gps.getValues()
                    vx,vy = vid2xy[req_vid]
                    myd = math.hypot(vx-x, vy-y)
                    if proposerd + 1e-6 < myd:
                        print(f"[YIELD] VID#{req_vid} to {proposer} (their {proposerd:.2f}m < mine {myd:.2f}m)")
                        # 回 ACK: accepted
                        r2r_send(tx, {"type":"handover_ack","vid":req_vid,"from":name,"accepted":True})
                        # 立即停车并退出（按你的指示）
                        send_vw(left_group, right_group, 0.0, 0.0)
                        return
                    else:
                        r2r_send(tx, {"type":"handover_ack","vid":req_vid,"from":name,"accepted":False})

            # handover_ack 这里对发起方仅做日志（不强依赖）
            elif t == "handover_ack":
                pass

        # 传感器
        x, y, _ = gps.getValues()
        yaw = get_yaw()

        # 到点判定
        if math.hypot(x - gx_fix, y - gy_fix) <= GOAL_TOL:
            send_vw(left_group, right_group, 0.0, 0.0)
            print(f"[DONE ] {name} reached goal vicinity at {fmt_xy(x,y)} (VID#{my_vid})")
            # 上报 supervisor
            notify_supervisor_victim_reached(robot, sup_tx, name, (x, y, 0.0), (gx_fix, gy_fix))
            # 广播完成
            r2r_send(tx, {"type":"victim_done","vid":int(my_vid),"robot_id":name})
            done_vids.add(my_vid)

            # —— 选下一个最近的未完成受害者，并尝试“协商接管” ——
            next_vid = try_pick_next_vid(x, y)
            if next_vid is None:
                print("[INFO ] no remaining victim, mission complete.")
                break

            # 向“原本负责该 VID 的机器人”发起接管请求
            owner = victim_owner.get(next_vid, None)
            vx, vy = vid2xy[next_vid]
            myd = math.hypot(vx-x, vy-y)
            r2r_send(tx, {"type":"handover_request","vid":int(next_vid),
                          "proposer":name, "proposer_dist":float(myd)})

            # 给对方一点点时间处理 ACK（不阻塞太久）
            t_ack_deadline = robot.getTime() + 0.5
            while robot.getTime() < t_ack_deadline and robot.step(step) != -1:
                for m in poll_r2r(robot, rx):
                    if isinstance(m, dict) and m.get("type") == "handover_ack" and int(m.get("vid",-1)) == next_vid:
                        print(f"[ACK  ] from {m.get('from')} accepted={m.get('accepted')}")
                pass

            # 切换到新 VID 并重新规划
            my_vid = next_vid
            gx_raw, gy_raw = vx, vy
            new_path, occ_raw, occ_plan, (gx_fix, gy_fix) = lightweight_refresh_and_replan(x, y, (gx_raw, gy_raw))
            if not new_path:
                print("[PLAN ] replan FAIL after switch, exit.")
                break
            path_xy = new_path
            acc_since_replan = 0.0
            print(f"[SWITCH] {name} now heading to VID#{my_vid} {fmt_xy(gx_raw, gy_raw)}")
            continue

        # LQR 跟踪
        try:
            v_cmd, w_cmd, _info = tracker.compute_cmd(x, y, yaw, path_xy, (gx_fix, gy_fix))
        except TypeError:
            v_cmd, w_cmd = tracker.compute_cmd(x, y, yaw, path_xy, (gx_fix, gy_fix))
        send_vw(left_group, right_group, v_cmd, w_cmd)

        # 累计距离 & 前方阻塞
        acc_since_replan += math.hypot(x - last_x, y - last_y)
        last_x, last_y = x, y
        need_replan = False
        if acc_since_replan >= REPLAN_DIST_M:
            need_replan = True
        elif path_front_blocked(gm, occ_plan, path_xy, x, y, ahead_m=FRONT_BLOCK_AHEAD_M):
            need_replan = True

        if need_replan:
            send_vw(left_group, right_group, 0.0, 0.0)
            new_path, occ_raw, occ_plan, (gx_fix, gy_fix) = lightweight_refresh_and_replan(x, y, (gx_fix, gy_fix))
            if new_path:
                path_xy = new_path
                print(f"[REPLAN] {name} new path len={len(path_xy)}")
            else:
                print(f"[REPLAN] {name} keep old path (refresh failed)")
            acc_since_replan = 0.0

    # ====== ⑥ 可视化落盘 ======
    P_final = logistic_safe(gm.L)
    occ_final = (P_final > OCC_THRESH).astype(np.uint8)
    scat = [([sx], [sy], dict(s=35, marker='o', label='start')),
            ([gx_fix], [gy_fix], dict(s=70, marker='*', label='victim'))]
    lines = None
    if path_xy and len(path_xy) >= 2:
        xs = [p[0] for p in path_xy]; ys = [p[1] for p in path_xy]
        lines = [(xs, ys, dict(linewidth=2))]
    save_png_matrix(occ_final, [gm.xmin, gm.xmax, gm.ymin, gm.ymax],
                    os.path.join(outdir, f"occ_raw_{name}_{ts}.png"),
                    title=f"occ_raw (thr={OCC_THRESH}) {name} {ts}",
                    cmap='gray', vmin=0, vmax=1, scatter=scat, lines=lines,
                    annots=[(gx_fix, gy_fix, f"goal {fmt_xy(gx_fix,gy_fix)}",
                             dict(textcoords="offset points", xytext=(6,6), fontsize=9,
                                  ha='left', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                                  arrowprops=dict(arrowstyle='-')))])

    # 路径文件
    if path_xy and len(path_xy) >= 2:
        try:
            csv_path = os.path.join(outdir, f"path_{name}_final_{ts}.csv")
            json_path = os.path.join(outdir, f"path_{name}_final_{ts}.json")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("x,y\n")
                for x_, y_ in path_xy:
                    f.write(f"{x_:.6f},{y_:.6f}\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"path": [[float(x_), float(y_)] for x_, y_ in path_xy]},
                          f, ensure_ascii=False, indent=2)
            print(f"[SAVE ] path CSV/JSON → {os.path.basename(csv_path)}, {os.path.basename(json_path)}")
        except Exception as e:
            print(f"[WARN ] save path failed: {e}")

    # 停车
    left_group, right_group = try_get_wheels(robot)
    send_vw(left_group, right_group, 0.0, 0.0)
    while robot.step(step) != -1:
        break

if __name__ == "__main__":
    main()
