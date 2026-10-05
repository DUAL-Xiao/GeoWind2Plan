import numpy as np


def sample_start_goal_inset_border(
    sdf,
    *,
    z0,
    inset=5,
    clearance=10.0,
    min_chord_ratio=0.9,
    rng=None,
    max_tries=10000,
):
    """
    Sample start/goal on the inset border (distance==inset from outer border),
    with chord length guaranteed by a square formula and sdf clearance at endpoints.

    min_chord_ratio:
        Minimum chord length ratio relative to the inset-square side length.
        Example: 0.9 means chord_len >= 0.9 * side.
    """
    if rng is None:
        rng = np.random.default_rng()

    X, Z, Y = sdf.shape
    xmin, xmax = inset, X - 1 - inset
    ymin, ymax = inset, Y - 1 - inset
    if not (0 <= z0 < Z):
        raise ValueError("z0 out of bounds.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Inset too large.")

    # Centered rectangle (inset area) for intersection math.
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax, ay = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    # Use the inscribed square for the clean d_max(theta) formula.
    a = float(min(ax, ay))  # half side length
    side = 2.0 * a
    L_min = float(min_chord_ratio) * side

    for _ in range(max_tries):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        n = np.array([-u[1], u[0]], dtype=float)

        # Square formula (no chord rejection):
        #   d_max = a(|sin|+|cos|) - L_min*|sin*cos|
        s = abs(u[1])  # |sin(theta)|
        c = abs(u[0])  # |cos(theta)|
        d_max = a * (s + c) - L_min * (s * c)
        if d_max <= 0.0:
            continue
        d = float(rng.uniform(-d_max, d_max))

        # Intersect line with centered square |x|<=a, |y|<=a.
        p0 = d * n
        t_min, t_max = -1e18, 1e18
        for (p0i, ui, lo, hi) in (
            (p0[0], u[0], -a, a),
            (p0[1], u[1], -a, a),
        ):
            if abs(ui) < 1e-12:
                if p0i < lo or p0i > hi:
                    t_min, t_max = 1.0, 0.0
                    break
                continue
            t1 = (lo - p0i) / ui
            t2 = (hi - p0i) / ui
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

        if t_max <= t_min:
            continue

        p1 = p0 + t_min * u + np.array([cx, cy])
        p2 = p0 + t_max * u + np.array([cx, cy])
        # Snap to integer inset border (distance==inset from outer border).
        def _snap(p):
            x = int(np.clip(int(round(p[0])), xmin, xmax))
            y = int(np.clip(int(round(p[1])), ymin, ymax))
            dx = min(abs(x - xmin), abs(x - xmax))
            dy = min(abs(y - ymin), abs(y - ymax))
            if dx <= dy:
                x = xmin if abs(x - xmin) <= abs(x - xmax) else xmax
            else:
                y = ymin if abs(y - ymin) <= abs(y - ymax) else ymax
            return np.array([x, y], dtype=int)

        p1i = _snap(p1)
        p2i = _snap(p2)

        start = np.array([int(p1i[0]), int(z0), int(p1i[1])], dtype=int)
        goal = np.array([int(p2i[0]), int(z0), int(p2i[1])], dtype=int)

        if sdf[start[0], start[1], start[2]] <= clearance:
            continue
        if sdf[goal[0], goal[1], goal[2]] <= clearance:
            continue

        return start, goal

    raise RuntimeError("Failed to sample a valid pair.")
