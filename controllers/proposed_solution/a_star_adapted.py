# -*- coding: utf-8 -*-
"""
A* grid planner consistent with your (x,y) world coordinates:
- r(row) corresponds to y, c(col) corresponds to x
- world_to_rc: r = floor((y - YMIN)/RES), c = floor((x - XMIN)/RES)
- rc_to_world returns cell center (x,y)

Now supports an optional per-cell cost map:
    plan(occ, start_xy, goal_xy, cost=None, cost_weight=1.0)

- occ: HxW uint8/bool (1=occupied/blocked, 0=free)
- cost: HxW float (>=0), added to g-cost as: tentative = g + move_cost + cost_weight*cost[nr,nc]
- cost_weight: scalar multiplier for the cost map contribution
"""
import math
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import List, Tuple, Optional

import numpy as np

@dataclass(order=True)
class PQItem:
    f: float
    g: float = field(compare=False)
    r: int = field(compare=False)
    c: int = field(compare=False)
    parent: Optional[Tuple[int, int]] = field(compare=False, default=None)

# 8-connected neighbors (dr, dc, move_cost)
NEIGHBORS: List[Tuple[int, int, float]] = [
    (-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)), ( 1, 1, math.sqrt(2))
]

class AStarGrid:
    def __init__(self, XMIN: float, XMAX: float, YMIN: float, YMAX: float, RES: float):
        self.XMIN, self.XMAX = XMIN, XMAX
        self.YMIN, self.YMAX = YMIN, YMAX
        self.RES = RES
        self.H = int(math.ceil((YMAX - YMIN) / RES))
        self.W = int(math.ceil((XMAX - XMIN) / RES))

    def world_to_rc(self, wx: float, wy: float) -> Tuple[int, int]:
        c = int((wx - self.XMIN) / self.RES)
        r = int((wy - self.YMIN) / self.RES)
        return r, c

    def rc_to_world(self, r: int, c: int) -> Tuple[float, float]:
        wx = self.XMIN + (c + 0.5) * self.RES
        wy = self.YMIN + (r + 0.5) * self.RES
        return wx, wy

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    @staticmethod
    def heuristic(r0: int, c0: int, r1: int, c1: int) -> float:
        # Euclidean in grid-index space; consistent with move_cost above
        return math.hypot(r1 - r0, c1 - c0)

    def plan(
        self,
        occ: np.ndarray,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        cost: Optional[np.ndarray] = None,
        cost_weight: float = 1.0,
    ) -> list:
        """
        Returns: [(x,y), ...] from start to goal; [] if failed.
        Backward compatible: if cost is None → behaves like classic A* on occ.
        """
        assert occ.shape == (self.H, self.W), "occ grid shape mismatch"
        if cost is not None:
            assert cost.shape == (self.H, self.W), "cost grid shape mismatch"
            # Guard: negative/NaN -> treat as 0
            cost = np.maximum(0.0, np.nan_to_num(cost, nan=0.0))

        sr, sc = self.world_to_rc(*start_xy)
        gr, gc = self.world_to_rc(*goal_xy)

        if not (self.in_bounds(sr, sc) and self.in_bounds(gr, gc)):
            return []
        if occ[sr, sc] == 1 or occ[gr, gc] == 1:
            return []

        open_heap: List[PQItem] = []
        g_cost = np.full(occ.shape, np.inf, dtype=float)
        parent = np.full((*occ.shape, 2), -1, dtype=int)

        g_cost[sr, sc] = 0.0
        h0 = self.heuristic(sr, sc, gr, gc)
        heappush(open_heap, PQItem(f=h0, g=0.0, r=sr, c=sc, parent=None))

        closed = np.zeros_like(occ, dtype=bool)

        while open_heap:
            cur = heappop(open_heap)
            r, c = cur.r, cur.c

            if closed[r, c]:
                continue
            closed[r, c] = True

            if (r, c) == (gr, gc):
                # reconstruct r,c -> world x,y
                path_xy = [self.rc_to_world(r, c)]
                pr, pc = parent[r, c]
                while pr >= 0:
                    path_xy.append(self.rc_to_world(pr, pc))
                    pr, pc = parent[pr, pc]
                path_xy.reverse()  # start -> goal
                return path_xy

            for dr, dc, move_cost in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc):
                    continue
                if closed[nr, nc] or occ[nr, nc] == 1:
                    continue

                add_cost = 0.0
                if cost is not None:
                    add_cost = cost_weight * float(cost[nr, nc])

                tentative = g_cost[r, c] + move_cost + add_cost
                if tentative < g_cost[nr, nc]:
                    g_cost[nr, nc] = tentative
                    parent[nr, nc] = (r, c)
                    f = tentative + self.heuristic(nr, nc, gr, gc)
                    heappush(open_heap, PQItem(f=f, g=tentative, r=nr, c=nc, parent=(r, c)))

        return []  # fail
