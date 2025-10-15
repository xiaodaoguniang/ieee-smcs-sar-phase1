# controllers/proposed_solution/victim_detector_xy.py
# -*- coding: utf-8 -*-
"""
Depth-only 红色目标定位（X–Y 地面系）模块化版本：
- 提供高阶：spin_scan_and_log() —— 原地旋转，按角步采图、检测、深度测距、写 JSONL、可选聚类评估。
- 提供低阶：pixel_depth_to_world() —— 已知像素&深度&位姿 → 世界坐标 (wx, wy)。
- 其余工具函数都可独立复用。
"""

from controller import Robot
import math, os, json, glob
import numpy as np
import cv2

# ===== 默认参数（可在调用时覆盖） =====
DEFAULTS = dict(
    STEP_DEG=45.0,
    TOTAL_DEG=360.0,
    WHEEL_SPEED=3.0,
    SAFE_TIMEOUT=60.0,
    SAVE_QUALITY=100,
    HSV_LO_1=np.array([  0, 90, 70], dtype=np.uint8),
    HSV_HI_1=np.array([ 10,255,255], dtype=np.uint8),
    HSV_LO_2=np.array([170, 90, 70], dtype=np.uint8),
    HSV_HI_2=np.array([180,255,255], dtype=np.uint8),
    RED_DOM_DELTA=25,
    AREA_MIN=50,
    MIN_BBOX_H_FRAC=0.40,
    DEPTH_OK=(0.60, 8.00),
    # 相机相对于机体：+X 向前、+Y 向左（我们的世界约定：+X 前进，yaw 增大逆时针）
    CAM_TX=0.027,     # 相机前向平移（m）
    CAM_TY=0.0,       # 相机侧向平移（m，+Y 左）
    CAM_YAW_OFF=0.0   # 相机自身相对机体的偏航（rad）
)

# ===== 工具函数 =====
def is_finite(x): return x == x and abs(x) != float("inf")
def vec_finite(v):
    try: return all(is_finite(float(x)) for x in v)
    except Exception: return False

def wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi

def bgra_to_bgr(img_bytes, w, h):
    buf = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 4))
    return buf[:, :, :3]

def make_cam_model(img_w, hfov_rad):
    fx  = (img_w * 0.5) / math.tan(hfov_rad * 0.5)
    cx0 = (img_w - 1) / 2.0
    return fx, cx0

def bearing_from_cx(cx, fx, cx0):
    # 像素在右（cx>cx0）→ bearing 为负（符合“逆时针为正”的世界约定）
    return -math.atan((cx - cx0) / fx)

def red_mask_hsv(bgr, HSV_LO_1, HSV_HI_1, HSV_LO_2, HSV_HI_2, RED_DOM_DELTA):
    bgr_blur = cv2.GaussianBlur(bgr, (3,3), 0)
    hsv = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, HSV_LO_1, HSV_HI_1)
    m2 = cv2.inRange(hsv, HSV_LO_2, HSV_HI_2)
    mask_hsv = cv2.bitwise_or(m1, m2)
    b, g, r = cv2.split(bgr_blur)
    r16, g16, b16 = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    mask_rgb = ((r16 > g16 + RED_DOM_DELTA) & (r16 > b16 + RED_DOM_DELTA)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask_hsv, mask_rgb)
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def median_depth_on_mask(depth_img, mask, bbox):
    x, y, w, h = bbox
    H, W = depth_img.shape
    x0, x1 = max(x,0), min(x+w, W)
    y0, y1 = max(y,0), min(y+h, H)
    sub_d = depth_img[y0:y1, x0:x1]
    sub_m = (mask[y0:y1, x0:x1] > 0)
    vals = sub_d[sub_m]
    if vals.size >= 20:
        return float(np.nanmedian(vals))
    cx0 = int(x + 0.30*w); cx1 = int(x + 0.70*w)
    cy0 = int(y + 0.30*h); cy1 = int(y + 0.70*h)
    cx0, cy0 = max(cx0,0), max(cy0,0)
    cx1, cy1 = min(cx1,W), min(cy1,H)
    vals2 = depth_img[cy0:cy1, cx0:cx1].ravel()
    if vals2.size:
        return float(np.nanmedian(vals2))
    return float("nan")

def detect_largest_red(bgr, depth_img, depth_valid_rng, *, 
                       HSV_LO_1, HSV_HI_1, HSV_LO_2, HSV_HI_2,
                       RED_DOM_DELTA, AREA_MIN, MIN_BBOX_H_FRAC):
    H, W = bgr.shape[:2]
    dmin, dmax = depth_valid_rng
    mask = red_mask_hsv(bgr, HSV_LO_1, HSV_HI_1, HSV_LO_2, HSV_HI_2, RED_DOM_DELTA)
    stats = {"mask_pixels": int((mask > 0).sum()), "area": 0}
    if stats["mask_pixels"] < AREA_MIN:
        return None, mask, stats
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, stats
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    stats["area"] = area
    if area < AREA_MIN:
        return None, mask, stats
    x,y,w,h = cv2.boundingRect(c)
    if h < int(H * MIN_BBOX_H_FRAC):
        return None, mask, stats
    M = cv2.moments(c)
    if M["m00"] > 1e-6:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx, cy = x + w/2.0, y + h/2.0
    d_depth = median_depth_on_mask(depth_img, mask, (x,y,w,h))
    if not (np.isfinite(d_depth) and dmin <= d_depth <= dmax):
        d_depth = float("nan")
    det = {"bbox":[int(x),int(y),int(w),int(h)], "pixel":[cx,cy],
           "depth_m": float(d_depth), "h_frac": float(h/float(H))}
    return det, mask, stats

def write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        print(json.dumps(obj, ensure_ascii=False), file=f)

# ===== 核心换算：像素/深度 → 世界系 (wx, wy) =====
def pixel_depth_to_world(px, py, pz, yaw_r,  # 机体位姿（世界 X–Y）
                         cam_tx, cam_ty, cam_yaw_off,
                         bearing, dist):
    """
    输入：
      (px,py,pz): 机体在世界中的位置
      yaw_r     : 机体偏航（弧度，+逆时针）
      cam_tx/ty : 相机相对机体的平移（m），+X 前、+Y 左
      cam_yaw_off: 相机相对机体的偏航
      bearing   : 像素列相对相机光轴的方位角（弧度，右=负，左=正）
      dist      : 深度量测（m）
    输出：
      (wx, wy)  : 世界系 X–Y 平面坐标
    """
    wx_cam = px + cam_tx*math.cos(yaw_r) - cam_ty*math.sin(yaw_r)
    wy_cam = py + cam_tx*math.sin(yaw_r) + cam_ty*math.cos(yaw_r)
    ang_w  = yaw_r + cam_yaw_off + bearing
    wx = float(wx_cam + math.cos(ang_w) * dist)
    wy = float(wy_cam + math.sin(ang_w) * dist)
    return wx, wy

# ===== 聚类（可选） =====
def cluster_xy_points(jsonl_glob="shots_*/*victim_detections_*.jsonl",
                      depth_ok=(0.60,8.0),
                      min_mask_pixels=2000,
                      min_h_frac=0.5,
                      cluster_eps=0.60,
                      cluster_min_pts=2):
    pts = []
    paths = glob.glob(jsonl_glob)
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try: rec = json.loads(line)
                    except Exception: continue
                    if rec.get("source") != "depth": continue
                    if rec.get("mask_pixels", 0) < min_mask_pixels: continue
                    if rec.get("h_frac", 0.0) < min_h_frac: continue
                    d = rec.get("dist_m")
                    if not (isinstance(d,(int,float)) and np.isfinite(d) and (depth_ok[0] <= d <= depth_ok[1])):
                        continue
                    w = rec.get("world", {})
                    x, y = w.get("x"), w.get("y")
                    if isinstance(x,(int,float)) and isinstance(y,(int,float)) and np.isfinite(x) and np.isfinite(y):
                        pts.append((float(x), float(y)))
        except FileNotFoundError:
            pass
    pts = np.array(pts, dtype=np.float32)
    if len(pts) == 0: return [], pts
    N = len(pts)
    used = np.zeros(N, dtype=bool); clusters = []
    eps2 = cluster_eps*cluster_eps
    D2 = ((pts[:,None,:] - pts[None,:,:])**2).sum(axis=2)
    for i in range(N):
        if used[i]: continue
        neigh = np.where(D2[i] <= eps2)[0]
        if len(neigh) < cluster_min_pts: continue
        cluster = set(neigh.tolist()); frontier = list(neigh.tolist())
        used[i] = True
        while frontier:
            j = frontier.pop()
            if used[j]: continue
            used[j] = True
            neigh2 = np.where(D2[j] <= eps2)[0]
            if len(neigh2) >= cluster_min_pts:
                for k in neigh2:
                    if k not in cluster:
                        cluster.add(k); frontier.append(k)
        C = pts[list(cluster)]
        cx = float(np.median(C[:,0])); cy = float(np.median(C[:,1]))
        clusters.append({"center": (cx, cy), "count": len(C)})
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters, pts

# ===== 高阶：原地旋转采样、检测、日志（与原脚本等价） =====
def spin_scan_and_log(robot: Robot, **kwargs):
    """
    旋转一圈、按固定角步拍照 → 红色检测 + 深度测距 + 世界坐标换算 → 写 JSONL。
    返回：
      {"jsonl_path": str, "angles_hit": [...], "angles_miss":[...]}
    """
    p = dict(DEFAULTS); p.update(kwargs or {})
    rid = robot.getName()
    ts  = int(robot.getBasicTimeStep())

    # wheels
    fl = robot.getDevice("fl_wheel_joint"); fr = robot.getDevice("fr_wheel_joint")
    rl = robot.getDevice("rl_wheel_joint"); rr = robot.getDevice("rr_wheel_joint")
    for m in (fl, fr, rl, rr):
        m.setPosition(float("inf")); m.setVelocity(0.0)

    # camera
    cam_rgb   = robot.getDevice("camera rgb");   cam_rgb.enable(ts)
    cam_depth = robot.getDevice("camera depth"); cam_depth.enable(ts)
    W, H  = cam_rgb.getWidth(), cam_rgb.getHeight()
    HFOV  = cam_rgb.getFov()
    FX, CX0 = make_cam_model(W, HFOV)
    dmin, dmax = cam_depth.getMinRange(), cam_depth.getMaxRange()

    # GPS / IMU
    gps = robot.getDevice("gps"); gps.enable(ts)
    imu = robot.getDevice("imu inertial_unit"); imu.enable(ts)

    # out
    out_dir = f"shots_{rid}"
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"victim_detections_{rid}.jsonl")

    # start CW spin
    for m in (fl, rl): m.setVelocity(-p["WHEEL_SPEED"])
    for m in (fr, rr): m.setVelocity(+p["WHEEL_SPEED"])

    STEP_RAD = math.radians(p["STEP_DEG"])
    shots_per_rev = int(round(p["TOTAL_DEG"] / p["STEP_DEG"]))
    next_idx, prev_yaw, turned_sum = 1, None, 0.0
    dir_sign, signed_set = 1.0, False
    start_time = robot.getTime()

    angles_hit, angles_miss = [], []

    while robot.step(ts) != -1:
        if robot.getTime() - start_time > p["SAFE_TIMEOUT"]:
            print(f"[{rid}] timeout -> stop"); break

        rpy = imu.getRollPitchYaw()
        if not vec_finite(rpy):
            print(f"[{rid}] waiting IMU..."); continue
        cur_yaw = float(rpy[2])
        if prev_yaw is None:
            prev_yaw = cur_yaw; print(f"[{rid}] waiting first valid yaw..."); continue
        dyaw = wrap_pi(cur_yaw - prev_yaw); prev_yaw = cur_yaw
        if not signed_set and abs(dyaw) > 1e-6:
            dir_sign = 1.0 if dyaw > 0 else -1.0; signed_set = True
        turned_sum = max(0.0, turned_sum + dir_sign * dyaw)

        current_idx = int(turned_sum // STEP_RAD)
        while current_idx >= next_idx and next_idx <= shots_per_rev:
            angle_deg = int(round(next_idx * p["STEP_DEG"]))
            # pause
            for m in (fl, fr, rl, rr): m.setVelocity(0.0)
            robot.step(max(ts//2, 1))

            stamp_ms = int(robot.getTime()*1000)
            base = f"t{stamp_ms}_a{angle_deg:03d}"
            rgb_path = os.path.join(out_dir, f"{base}_rgb.png")
            cam_rgb.saveImage(rgb_path, p["SAVE_QUALITY"])

            bgr = bgra_to_bgr(cam_rgb.getImage(), W, H)
            dH, dW = cam_depth.getHeight(), cam_depth.getWidth()
            depth_raw = np.array(cam_depth.getRangeImage(), dtype=np.float32).reshape((dH, dW))
            depth_img = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST) if (dW, dH)!=(W, H) else depth_raw

            det, mask, stats = detect_largest_red(
                bgr, depth_img, (dmin, dmax),
                HSV_LO_1=p["HSV_LO_1"], HSV_HI_1=p["HSV_HI_1"],
                HSV_LO_2=p["HSV_LO_2"], HSV_HI_2=p["HSV_HI_2"],
                RED_DOM_DELTA=p["RED_DOM_DELTA"],
                AREA_MIN=p["AREA_MIN"],
                MIN_BBOX_H_FRAC=p["MIN_BBOX_H_FRAC"]
            )
            ok = False
            if det is not None:
                cx, cy   = det["pixel"]
                d_depth  = det["depth_m"]
                if np.isfinite(d_depth) and (p["DEPTH_OK"][0] <= d_depth <= p["DEPTH_OK"][1]):
                    bearing = bearing_from_cx(cx, FX, CX0)
                    px, py, pz = gps.getValues()
                    yaw_r = float(imu.getRollPitchYaw()[2])
                    wx, wy = pixel_depth_to_world(px, py, pz, yaw_r,
                                                  p["CAM_TX"], p["CAM_TY"], p["CAM_YAW_OFF"],
                                                  bearing, float(d_depth))
                    # 可视化小记号
                    x,y,w,h = det["bbox"]
                    vis = bgr.copy()
                    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0,255,0), -1)
                    cv2.putText(vis, f"depth:{d_depth:.2f}m", (x, max(0,y-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(out_dir, f"{base}_vis.png"), vis)

                    rec = {
                        "t": robot.getTime(),
                        "robot": rid,
                        "shot_angle_deg": angle_deg,
                        "pixel": [round(float(cx),1), round(float(cy),1)],
                        "bearing_deg": round(math.degrees(bearing), 2),
                        "dist_m": round(float(d_depth), 3),
                        "source": "depth",
                        "world_plane": "xy",
                        "world": {"x": wx, "y": wy},
                        "pose": {
                            "px": px, "py": py, "pz": pz,
                            "yaw_deg": round(math.degrees(yaw_r),2),
                            "ang_w_deg": round(math.degrees(yaw_r + p['CAM_YAW_OFF'] + bearing),2)
                        },
                        "rgb_path": rgb_path,
                        "bbox": det["bbox"],
                        "h_frac": round(float(det["h_frac"]), 3),
                        "cam_fov_deg": round(math.degrees(HFOV),1),
                        "mask_pixels": int(stats.get("mask_pixels", 0))
                    }
                    write_jsonl(jsonl_path, rec)
                    print(f"[{rid}] ✔ DETECT@{angle_deg:03d}° depth={d_depth:.2f}m "
                          f"bear={math.degrees(bearing):+.1f}° world≈({wx:+.2f},{wy:+.2f})")
                    ok = True
                else:
                    cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask)
                    print(f"[{rid}] △ RED@{angle_deg:03d}° but NO valid depth (mask_pixels={stats.get('mask_pixels',0)})")
            else:
                cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask)
                print(f"[{rid}] ✗ NO-RED@{angle_deg:03d}° (mask_pixels={stats.get('mask_pixels',0) if isinstance(stats,dict) else 0})")

            if ok: angles_hit.append(angle_deg)
            else:  angles_miss.append(angle_deg)

            # resume spin
            next_idx += 1
            if next_idx <= shots_per_rev:
                for m in (fl, rl): m.setVelocity(-p["WHEEL_SPEED"])
                for m in (fr, rr): m.setVelocity(+p["WHEEL_SPEED"])

        if next_idx > shots_per_rev:
            break

    # stop
    for m in (fl, fr, rl, rr): m.setVelocity(0.0)

    # summary
    angles_hit.sort(); angles_miss.sort()
    with open(os.path.join(f"shots_{rid}", f"summary_{rid}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "robot": rid,
            "step_deg": p["STEP_DEG"],
            "shots_total": int(round(p["TOTAL_DEG"]/p["STEP_DEG"])),
            "angles_hit": angles_hit,
            "angles_miss": angles_miss
        }, f, ensure_ascii=False, indent=2)

    return {"jsonl_path": jsonl_path, "angles_hit": angles_hit, "angles_miss": angles_miss}
