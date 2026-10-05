import numpy as np
import random

class RRTPlanner3D:
    """
    Rapidly-exploring Random Tree (RRT) planner for 3D occupancy grids,
    with SDF-based adaptive step sizing and optional post-processing:
    - Randomized shortcutting
    - Elastic band smoothing using potential fields to ensure collision-free smoothing
    """
    def __init__(
        self,
        occupancy,
        sdf,
        step_size=1.0,
        min_step_size=0.1,
        sdf_scale=1.0,
        max_iters=5000,
        goal_sample_rate=0.1,
        shortcut_iters=100,
        shortcut_max_span=None,
        smooth_iters=50,
        el_gain=0.2,
        rep_gain=0.1,
        repulsive_radius=None,
        clearance=0.0,
    ):
        # Validate inputs
        assert sdf.shape == occupancy.shape, "SDF and occupancy grid must have same shape"
        self.occupancy = occupancy
        self.sdf = sdf
        self.step_size = step_size
        self.min_step_size = min_step_size
        self.sdf_scale = sdf_scale
        self.max_iters = max_iters
        self.goal_sample_rate = goal_sample_rate
        self.shortcut_iters = shortcut_iters
        self.shortcut_max_span = shortcut_max_span
        # Smoothing parameters
        self.smooth_iters = smooth_iters
        self.el_gain = el_gain
        self.rep_gain = rep_gain
        # repulsive radius: within this distance, repulsion acts
        if repulsive_radius is None:
            repulsive_radius = step_size
        self.repulsive_radius = repulsive_radius
        # Minimum SDF distance required to be considered collision-free.
        self.clearance = clearance

        self.x_limit, self.y_limit, self.z_limit = occupancy.shape

    def plan(self, start, goal):
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        if self._is_in_obstacle(start) or self._is_in_obstacle(goal):
            raise ValueError("Start or goal is inside an obstacle.")

        nodes = [start]
        parents = {0: None}

        for _ in range(self.max_iters):
            rnd = goal if np.random.rand() < self.goal_sample_rate else self._sample_free()
            idx_near = self._nearest(nodes, rnd)
            q_new = self._steer(nodes[idx_near], rnd)
            if not self._collision_free(nodes[idx_near], q_new):
                continue

            nodes.append(q_new)
            parents[len(nodes) - 1] = idx_near

            if np.linalg.norm(q_new - goal) <= self.step_size and self._collision_free(q_new, goal):
                raw_path = self._reconstruct_path(nodes, parents)
                if np.linalg.norm(raw_path[-1] - goal) > 0:
                    raw_path.append(goal)
                path = self._shortcut_path(raw_path, self.shortcut_iters)
                path = self._elastic_smooth(path)
                return path

        return None

    def _sample_free(self):
        while True:
            p = np.random.uniform((0, 0, 0), (self.x_limit, self.y_limit, self.z_limit))
            if not self._is_in_obstacle(p):
                return p

    def _nearest(self, nodes, q):
        pts = np.array(nodes)
        d2 = np.sum((pts - q)**2, axis=1)
        return int(np.argmin(d2))

    def _steer(self, q_from, q_to):
        vec = q_to - q_from
        dist = np.linalg.norm(vec)
        idx = np.floor(q_from).astype(int)
        np.clip(idx, [0, 0, 0], [self.x_limit-1, self.y_limit-1, self.z_limit-1], out=idx)
        local_dist = self.sdf[idx[0], idx[1], idx[2]]
        d_step = np.clip(local_dist * self.sdf_scale, self.min_step_size, self.step_size)
        if dist <= d_step:
            return q_to
        return q_from + (vec / dist) * d_step

    def _collision_free(self, p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(int(np.ceil(dist * 2)), 1)
        t = np.linspace(0, 1, steps)
        pts = p1[None, :] + t[:, None] * (p2 - p1)
        idx = np.floor(pts).astype(int)
        np.clip(idx, [0,0,0], [self.x_limit-1, self.y_limit-1, self.z_limit-1], out=idx)
        for x,y,z in {tuple(i) for i in idx}:
            if self.occupancy[x, y, z]:
                return False
            if self.sdf[x, y, z] <= self.clearance:
                return False
        return True

    def _is_in_obstacle(self, p):
        x, y, z = np.floor(p).astype(int)
        if not (0 <= x < self.x_limit and 0 <= y < self.y_limit and 0 <= z < self.z_limit):
            return True
        if self.occupancy[x, y, z]:
            return True
        return self.sdf[x, y, z] <= self.clearance

    def _reconstruct_path(self, nodes, parents):
        path, idx = [], len(nodes)-1
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path[::-1]

    def _shortcut_path(self, path, iterations):
        n = len(path)
        if n < 3:
            return path
        for _ in range(iterations):
            n = len(path)
            i = random.randint(0, n-3)
            span = self.shortcut_max_span or (n - i - 1)
            max_j = min(n-1, i + span)
            if max_j <= i+1:
                continue
            j = random.randint(i+2, max_j)
            if self._collision_free(path[i], path[j]):
                path = path[:i+1] + path[j:]
        return path

    def _elastic_smooth(self, path):
        """
        Elastic band smoothing: apply spring and repulsive forces based on SDF gradient,
        ensuring collision-free updates.
        """
        path = [np.array(p, dtype=float) for p in path]
        for _ in range(self.smooth_iters):
            for i in range(1, len(path)-1):
                p_prev, p, p_next = path[i-1], path[i], path[i+1]
                # spring force
                f_el = self.el_gain * (p_prev + p_next - 2*p)
                # repulsive force with protection
                idx = np.floor(p).astype(int)
                np.clip(idx, [0,0,0], [self.x_limit-1, self.y_limit-1, self.z_limit-1], out=idx)
                dist_to_obs = self.sdf[idx[0], idx[1], idx[2]]
                grad = self._sdf_gradient(idx)
                eps = 1e-3
                d = max(dist_to_obs, eps)
                if d < self.repulsive_radius:
                    mag = self.rep_gain * (1.0/d - 1.0/self.repulsive_radius)
                    f_rep = mag * grad
                else:
                    f_rep = np.zeros(3)
                p_new = p + f_el + f_rep
                # clamp within bounds
                p_new = np.clip(p_new, [0,0,0], [self.x_limit, self.y_limit, self.z_limit])
                # collision check
                if self._collision_free(p_prev, p_new) and self._collision_free(p_new, p_next):
                    path[i] = p_new
        return [p.tolist() for p in path]

    def _sdf_gradient(self, idx):
        x,y,z = idx
        # central difference
        gx = (self.sdf[min(x+1, self.x_limit-1), y, z] - self.sdf[max(x-1, 0), y, z]) * 0.5
        gy = (self.sdf[x, min(y+1, self.y_limit-1), z] - self.sdf[x, max(y-1, 0), z]) * 0.5
        gz = (self.sdf[x, y, min(z+1, self.z_limit-1)] - self.sdf[x, y, max(z-1, 0)]) * 0.5
        grad = np.array([gx, gy, gz])
        norm = np.linalg.norm(grad)
        if norm > 0:
            grad /= norm
        return grad


# ---------------------------------------------------------------------------
# Corridor-aware RRT planner
# This subclass adds an optional horizontal-corridor constraint defined by
# the start-goal line in the XY horizontal plane (dimensions 0 and 2 of the
# grid, because dimension 1 is the height/Z axis).
#
# Corridor total width = corridor_alpha * ||goal - start||_3D (grid units).
# Half-width d_max = 0.5 * corridor_alpha * ||goal - start||_3D.
# If corridor_alpha < 0, plan() delegates to RRTPlanner3D.plan() without any
# corridor constraint.
# ---------------------------------------------------------------------------
class RRTPlanner3DCorridor(RRTPlanner3D):
    """
    RRT planner with an optional horizontal corridor constraint.

    The corridor is centred on the straight line between start and goal
    projected onto the horizontal plane (axes 0 and 2 in grid index space;
    axis 1 is the height axis).

    Parameters
    ----------
    All constructor parameters are identical to RRTPlanner3D.
    The corridor width is supplied per-call via plan(..., corridor_alpha=...).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, start, goal, corridor_alpha: float = -1.0):
        """
        Plan a path from start to goal.

        Parameters
        ----------
        start, goal : array-like of shape (3,)
            3-D grid-index coordinates [x, height, y].
        corridor_alpha : float
            If < 0 → no corridor constraint (identical to RRTPlanner3D.plan).
            Otherwise, corridor total width = corridor_alpha * 3-D Euclidean
            distance between start and goal (in grid units).
        """
        if corridor_alpha < 0.0:
            return super().plan(start, goal)

        start = np.array(start, dtype=float)
        goal  = np.array(goal,  dtype=float)

        if self._is_in_obstacle(start) or self._is_in_obstacle(goal):
            raise ValueError("Start or goal is inside an obstacle.")

        # --- Corridor geometry (horizontal plane: axes 0 and 2) ---
        # S, G are 2-D horizontal projections.
        S = np.array([start[0], start[2]], dtype=float)
        G = np.array([goal[0],  goal[2]],  dtype=float)
        d = G - S                                  # 2-D direction vector
        d_norm_sq = float(np.dot(d, d))            # ||d||^2 in 2-D (grid units^2)

        dist_3d = float(np.linalg.norm(goal - start))  # 3-D Euclidean (grid units)
        d_max   = 0.5 * corridor_alpha * dist_3d        # half-width
        d_max_sq = d_max ** 2

        # If start and goal coincide horizontally, the corridor direction is
        # undefined → skip the constraint (treat as no corridor).
        use_corridor = d_norm_sq > 1e-12

        # Pre-compute corridor local frame for efficient direct sampling.
        # Avoids O(1/alpha) rejection rate of sampling in the full 3-D box.
        if use_corridor:
            d_norm_horiz = np.sqrt(d_norm_sq)
            d_hat  = d / d_norm_horiz                       # unit vector along corridor (2-D horiz)
            d_perp = np.array([-d_hat[1], d_hat[0]])        # perpendicular horizontal unit vector
            L_horiz = d_norm_horiz                           # projected start→goal length
        else:
            d_hat   = np.array([1.0, 0.0])
            d_perp  = np.array([0.0, 1.0])
            L_horiz = 0.0

        def _in_corridor(p) -> bool:
            """Return True if grid point p satisfies the corridor constraint."""
            if not use_corridor:
                return True
            P = np.array([p[0], p[2]], dtype=float)
            # 2-D signed cross product: (P-S) × d  (scalar)
            cross = (P[0] - S[0]) * d[1] - (P[1] - S[1]) * d[0]
            # Equivalent to: (|cross| / ||d||) <= d_max
            return d_max_sq * d_norm_sq - cross * cross >= 0.0

        # --- RRT main loop ---
        nodes   = [start]
        parents = {0: None}

        for _ in range(self.max_iters):
            # Sample: bias toward goal or sample inside corridor
            if np.random.rand() < self.goal_sample_rate:
                rnd = goal
            else:
                rnd = self._sample_free_corridor(S, d_hat, d_perp, d_max, L_horiz)

            idx_near = self._nearest(nodes, rnd)
            q_new    = self._steer(nodes[idx_near], rnd)

            # Standard collision check (occupancy + SDF clearance)
            if not self._collision_free(nodes[idx_near], q_new):
                continue
            # Additional corridor constraint on the new node
            if not _in_corridor(q_new):
                continue

            nodes.append(q_new)
            parents[len(nodes) - 1] = idx_near

            if (np.linalg.norm(q_new - goal) <= self.step_size
                    and self._collision_free(q_new, goal)):
                raw_path = self._reconstruct_path(nodes, parents)
                if np.linalg.norm(raw_path[-1] - goal) > 0:
                    raw_path.append(goal)
                # NOTE: _shortcut_path and _elastic_smooth are inherited.
                # They do not re-check the corridor constraint.  In the
                # current calling convention shortcut_iters=0 and
                # smooth_iters=0, so they are no-ops and the corridor is
                # preserved.  If you enable them, consider overriding these
                # methods to also enforce the corridor.
                path = self._shortcut_path(raw_path, self.shortcut_iters)
                path = self._elastic_smooth(path)
                return path

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _sample_free_corridor(self, S, d_hat, d_perp, d_max, L_horiz,
                               max_tries: int = 5000):
        """
        Sample a collision-free point directly in the corridor coordinate
        frame, eliminating the O(1/alpha) inefficiency of rejection sampling
        in the full 3-D bounding box.

        Corridor local frame (2-D horizontal):
            along  : S + t * d_hat,   t  ∈ [-step_size, L_horiz + step_size]
            lateral: + lat * d_perp,  lat ∈ [-d_max, d_max]
        Height (grid axis 1) is sampled freely over the full valid range.

        Every generated point is inside the corridor by construction, so no
        corridor rejection test is needed.  The only rejections are grid-
        bounds and obstacle checks, whose acceptance rate is independent of
        alpha.  This makes very narrow corridors (small alpha) just as
        efficient to sample as wide ones.
        """
        delta = float(self.step_size)   # small margin beyond the S–G endpoints
        t_lo  = -delta
        t_hi  =  L_horiz + delta

        for _ in range(max_tries):
            t   = np.random.uniform(t_lo, t_hi)
            lat = np.random.uniform(-d_max, d_max)
            # Sample height strictly within valid index range so that downstream
            # np.round(P_grid) indexing cannot produce an out-of-bounds z index.
            h_hi = max(0.0, float(self.y_limit) - 1.0)
            h   = np.random.uniform(0.0, h_hi)

            px = S[0] + t * d_hat[0] + lat * d_perp[0]
            py = S[1] + t * d_hat[1] + lat * d_perp[1]

            # Cheap grid-bounds check before the costlier obstacle lookup
            if not (0.0 <= px < self.x_limit and 0.0 <= py < self.z_limit):
                continue
            p = np.array([px, h, py])
            if not self._is_in_obstacle(p):
                return p

        # Fallback: corridor is entirely blocked (extremely rare)
        return super()._sample_free()

