Webots SAR Controller — README
================================

Multi-robot search-&-assist baseline with depth-only victim detection, LiDAR grid mapping, A* planning, and LQR path tracking.

Features
--------
- Depth-only red-object (“victim”) detection + world-XY projection with a one-revolution scan logger. Outputs JSONL + images for later clustering.
- 2D log-odds grid mapping from LiDAR (Webots-angle aligned), with Bresenham ray updates.
- A* on an occupancy grid with optional soft-cost; consistent world↔grid conversions.
- Differential-drive LQR-like tracker with dynamic lookahead, goal tolerance, and velocity caps.
- End-to-end controller wiring: spin-scan → cluster victims → plan on grid → track path → periodic lightweight replan.

Repo Layout (controllers/proposed_solution)
-------------------------------------------
- proposed_solution.py — main Webots controller (mapping, planning, tracking, comms). Key tunables: map bounds/res, occupancy threshold, replan triggers, robot radius.
- victim_detector_xy.py — spin-scan, red detection, depth-to-world, clustering utilities.
- lidar_gridmap.py — GridMap (log-odds), Webots-angle helper, logistic conversion.
- a_star_adapted.py — AStarGrid with world↔rc helpers and cost-aware planning.
- lqr_tracker.py — LQRTracker.compute_cmd() returning (v, w, info).
- import_smoke_test.py — quick import self-check for PythonRobotics modules on sys.path.

Quick Start
-----------
1) Environment
   - Webots world with devices: gps, imu/inertial_unit (or compass), laser, wheel motors (fl_/fr_/rl_/rr_wheel_joint), cameras (camera rgb, camera depth).
   - Python deps: numpy, opencv-python, matplotlib (only for saved figures).

2) Sanity Check (optional)
   - Run:  python controllers/proposed_solution/import_smoke_test.py
   - Prints OK/FAIL per module and the appended third_party/PythonRobotics path.

3) Launch in Webots
   - Assign controllers/proposed_solution/proposed_solution.py to the robot and run.
   - On start it logs LiDAR info, map bounds, and robot radius; then performs vision spin-scan.

How It Works (Pipeline)
-----------------------
1. Spin-scan & Detect
   The robot rotates in place, pausing at fixed angle steps to capture RGB+Depth, detect the largest valid red region, and estimate a depth-validated world-XY for each hit; results appended to shots_<RobotName>/victim_detections_*.jsonl.

2. Cluster Victim Detections
   JSONL records are filtered (mask size, bbox height, depth range) and clustered; cluster medians form candidate victim centers.

3. Map Building
   LiDAR scans are fused into a log-odds grid using Webots-aligned angles; binary occupancy derives via thresholding (default 0.65).

4. Path Planning
   A* plans from snapped-to-free start/goal in the inflated occupancy map; replans when moving or if near-front blocking is detected.

5. Path Tracking
   LQR-style controller outputs (v, w) with dynamic lookahead and slowdown near goal; arrival/tail flags available in info.

6. Artifacts
   Final occupancy PNG and path CSV/JSON are saved under map_out/.

Key Parameters
--------------
Before running, please make sure to specify the world size (XMIN/XMAX/YMIN/YMAX) and the number of victims to be searched (e.g., MAX_VICTIMS or an expected upper limit).

- Map window & resolution: XMIN/XMAX/YMIN/YMAX, RES (m/cell).
- Occupancy threshold: OCC_THRESH (default 0.65).
- Robot geometry: ROBOT_RADIUS, EXTRA_MARGIN for inflation.
- Replanning cadence: REPLAN_DIST_M, REPLAN_SPINS (or a fixed period).
- Tracker caps: MAX_V, MAX_W.
- Victim cap: MAX_VICTIMS and detection/cluster thresholds in victim_detector_xy.py::DEFAULTS.

Typical Outputs
---------------
- shots_<RobotName>/*_rgb.png — paused frames during spin-scan.
- shots_<RobotName>/victim_detections_*.jsonl — one JSON per detection (pixel, depth, world XY, bbox & stats).
- map_out/occ_raw_*.png — final occupancy (0/1) with overlays; map_out/path_*.csv|json.

Notes & Conventions
-------------------
- World plane is (X,Y); yaw is CCW-positive about +Z (up). Webots LiDAR range image indexes left→right correspond to CW angles; typical usage: th = yaw - angle (ccw=False in helpers).
- Planner/mapper index math: world_to_rc() maps (wx, wy) to row/col consistently across GridMap and AStarGrid.

License
-------
This controller integrates third-party modules under their respective licenses (e.g., PythonRobotics) and is intended for research/education use. Check third-party LICENSE files before redistribution.
