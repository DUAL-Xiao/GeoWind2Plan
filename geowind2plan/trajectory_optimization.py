import numpy as np
import casadi as ca


def _build_feasible_initial_guess(path_phys, dt0, v_max, a_max, max_iter=20):
    """
    Build v/a to satisfy trapezoidal kinematics exactly.
    If limits are violated, increase dt and retry.
    """
    n = path_phys.shape[0]
    d = path_phys[1:] - path_phys[:-1]

    def _build_va(dt_val):
        v_init = np.zeros_like(path_phys)
        if n > 1:
            v_init[0] = 2.0 * d[0] / dt_val
            for i in range(n - 1):
                v_init[i + 1] = 2.0 * d[i] / dt_val - v_init[i]
        a_init = np.zeros_like(path_phys)
        if n > 2:
            a_init[0] = 2.0 * (v_init[1] - v_init[0]) / dt_val
            for i in range(n - 1):
                a_init[i + 1] = 2.0 * (v_init[i + 1] - v_init[i]) / dt_val - a_init[i]
        elif n > 1:
            a_init[0] = 2.0 * (v_init[1] - v_init[0]) / dt_val
            a_init[1] = a_init[0]
        return v_init, a_init

    dt_val = float(dt0)
    for _ in range(max_iter):
        v_init, a_init = _build_va(dt_val)
        if n <= 1:
            return v_init, a_init, dt_val
        v_ok = np.max(np.linalg.norm(v_init, axis=1)) <= v_max
        a_ok = np.max(np.linalg.norm(a_init, axis=1)) <= a_max
        if v_ok and a_ok:
            return v_init, a_init, dt_val
        dt_val *= 1.5
    return v_init, a_init, dt_val


def _smooth_norm(vec, eps=1e-9):
    """Smooth norm to avoid NaNs at zero."""
    return ca.sqrt(ca.dot(vec, vec) + eps)


def build_interpolants(sdf_grid, wind_grid, grid_scale):
    """
    Build trilinear interpolants for the SDF and the wind field.
    Coordinates are in physical units; grid_scale maps grid index -> physical.
    """
    grid_scale = np.asarray(grid_scale, dtype=float)
    x_axis = np.arange(sdf_grid.shape[0], dtype=float).tolist()
    z_axis = np.arange(sdf_grid.shape[1], dtype=float).tolist()
    y_axis = np.arange(sdf_grid.shape[2], dtype=float).tolist()

    sdf_vals = np.asarray(sdf_grid, dtype=float).ravel(order="F").tolist()
    wx_vals = np.asarray(wind_grid[..., 0], dtype=float).ravel(order="F").tolist()
    wy_vals = np.asarray(wind_grid[..., 1], dtype=float).ravel(order="F").tolist()
    wz_vals = np.asarray(wind_grid[..., 2], dtype=float).ravel(order="F").tolist()

    sdf_lin = ca.interpolant("sdf_linear", "linear", [x_axis, z_axis, y_axis], sdf_vals)
    wx_lin = ca.interpolant("wx_linear", "linear", [x_axis, z_axis, y_axis], wx_vals)
    wy_lin = ca.interpolant("wy_linear", "linear", [x_axis, z_axis, y_axis], wy_vals)
    wz_lin = ca.interpolant("wz_linear", "linear", [x_axis, z_axis, y_axis], wz_vals)

    p = ca.MX.sym("p", 3)
    p_grid = p / grid_scale

    sdf_interp = ca.Function("sdf_interp", [p], [sdf_lin(p_grid)])
    wind_interp = ca.Function(
        "wind_interp",
        [p],
        [ca.vertcat(wx_lin(p_grid), wy_lin(p_grid), wz_lin(p_grid))],
    )

    return sdf_interp, wind_interp


def optimize_path_casadi(
    path_init,
    sdf_fun,
    wind_fun,
    clearance,
    v_max,
    a_max,
    t_max,
    dt0,
    map_shape,
    grid_scale=(4.0, 1.5, 4.0),
    dt_min=1e-3,
    dt_max=500.0,
    seg_samples=2,
    max_iter=200,
    verbose=True,
):
    """
    Optimize path energy with CasADi.
    Variables: x, v, a (all in physical units) and uniform dt.
    Constraints: kinematics, collision clearance, speed/accel/thrust limits.
    seg_samples controls how many interior samples per segment must be collision-free.
    If wind_fun is None, wind is treated as zero.
    """
    grid_scale = np.asarray(grid_scale, dtype=float)
    map_shape = np.asarray(map_shape, dtype=float)
    path_init = np.asarray(path_init, dtype=float)
    path_phys = path_init * grid_scale
    n = path_phys.shape[0]
    start_phys = path_phys[0]
    goal_phys = path_phys[-1]

    x = ca.MX.sym("x", n, 3)
    v = ca.MX.sym("v", n, 3)
    a = ca.MX.sym("a", n, 3)
    dt = ca.MX.sym("dt")

    # --- Parameters for energy model ---
    rho = 1.18
    Cd = 1.0
    Aframe = 0.16
    m = 5.0
    A = 0.5
    P_p_hover = 100.0
    g = 9.81
    g_vec = ca.DM([0.0, -g, 0.0])
    eps = 1e-9

    g_list = []
    lbg = []
    ubg = []
    power = []

    # Kinematic constraints
    for i in range(n - 1):
        xi = x[i, :].T
        xip1 = x[i + 1, :].T
        vi = v[i, :].T
        vip1 = v[i + 1, :].T
        ai = a[i, :].T
        aip1 = a[i + 1, :].T
        x_next = xi + 0.5 * (vi + vip1) * dt
        v_next = vi + 0.5 * (ai + aip1) * dt
        g_list.append(xip1 - x_next)
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]
        g_list.append(vip1 - v_next)
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]

    # Boundary position constraints (x0 = start, xN = goal)
    g_list.append(x[0, :].T - start_phys)
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]
    g_list.append(x[-1, :].T - goal_phys)
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]

    # Pointwise constraints and power terms
    for i in range(n):
        xi = x[i, :].T
        vi = v[i, :].T
        ai = a[i, :].T

        # Collision clearance
        g_list.append(sdf_fun(xi) - clearance)
        lbg.append(0.0)
        ubg.append(ca.inf)

        # Acceleration limit
        g_list.append(a_max - _smooth_norm(ai))
        lbg.append(0.0)
        ubg.append(ca.inf)

        # Aerodynamics for thrust constraint and energy
        if wind_fun is None:
            v_w = ca.MX.zeros(3, 1)
        else:
            v_w = wind_fun(xi)
        v_a = vi - v_w

        # Airspeed limit (constraint on relative-to-air velocity)
        g_list.append(v_max - _smooth_norm(v_a))
        lbg.append(0.0)
        ubg.append(ca.inf)

        v_a_norm = ca.sqrt(ca.dot(v_a, v_a) + eps)
        D = -0.5 * rho * Cd * Aframe * (v_a_norm * v_a)

        T_vec = m * ai - m * g_vec - D
        T_norm = _smooth_norm(T_vec)

        # Thrust limit
        g_list.append(t_max - T_norm)
        lbg.append(0.0)
        ubg.append(ca.inf)

        t_hat = T_vec / (T_norm + eps)
        v_c = ca.dot(v_a, t_hat)
        v_i = -0.5 * v_c + ca.sqrt((0.5 * v_c) ** 2 + T_norm / (2.0 * rho * A) + eps)

        P_useful = ca.dot(T_vec, v_a)
        P_induced = T_norm * v_i
        P_p = P_p_hover * (T_norm / (m * g)) ** 1.5
        power.append(P_useful + P_induced + P_p)

    # Segment interior collision checks (linear interpolation)
    if seg_samples > 0:
        for i in range(n - 1):
            xi = x[i, :].T
            xip1 = x[i + 1, :].T
            for k in range(1, seg_samples + 1):
                alpha = float(k) / float(seg_samples + 1)
                x_seg = (1.0 - alpha) * xi + alpha * xip1
                g_list.append(sdf_fun(x_seg) - clearance)
                lbg.append(0.0)
                ubg.append(ca.inf)

    # Total energy (trapezoidal integration)
    E_total = 0
    for i in range(n - 1):
        E_total += 0.5 * (power[i] + power[i + 1]) * dt

    # Decision vector
    z = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(v, -1, 1), ca.reshape(a, -1, 1), dt)
    g = ca.vertcat(*g_list)

    # Initial guess
    v0, a0, dt0_eff = _build_feasible_initial_guess(path_phys, dt0, v_max, a_max)
    z0 = np.concatenate(
        [
            path_phys.reshape(-1, order="F"),
            v0.reshape(-1, order="F"),
            a0.reshape(-1, order="F"),
            [dt0_eff],
        ]
    )

    # Variable bounds (keep dt bounded; others free)
    lbx = -ca.inf * np.ones(z.shape[0])
    ubx = ca.inf * np.ones(z.shape[0])
    # Position bounds in physical coordinates.
    map_upper = (map_shape - 1.0) * grid_scale
    x_lower = np.zeros(n * 3)
    x_upper = np.concatenate(
        [
            np.full(n, map_upper[0]),
            np.full(n, map_upper[1]),
            np.full(n, map_upper[2]),
        ]
    )
    lbx[: n * 3] = x_lower
    ubx[: n * 3] = x_upper
    lbx[-1] = dt_min
    ubx[-1] = dt_max

    nlp = {"x": z, "f": E_total, "g": g}
    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "print_time": 1 if verbose else 0,
        "ipopt.max_iter": int(max_iter),
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    try:
        sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    except RuntimeError as exc:
        return None, None, {"success": False, "message": str(exc)}

    z_opt = sol["x"].full().squeeze()
    x_opt = z_opt[: n * 3].reshape(n, 3, order="F")
    v_opt = z_opt[n * 3 : n * 6].reshape(n, 3, order="F")
    a_opt = z_opt[n * 6 : n * 9].reshape(n, 3, order="F")
    dt_opt = float(z_opt[-1])

    path_opt = x_opt / grid_scale
    info = {
        "success": True,
        "status": solver.stats()["return_status"],
        "iterations": solver.stats().get("iter_count", None),
        "dt": dt_opt,
        "v_opt": v_opt,
        "a_opt": a_opt,
        "obj_f": float(sol["f"]),
    }

    return path_opt, float(sol["f"]), info


# ---------------------------------------------------------------------------
# Corridor-aware trajectory optimizer
# This function adds an optional horizontal-corridor constraint defined by
# the straight line between start and goal projected onto the horizontal
# plane (physical coordinate axes 0 and 2; axis 1 is the height axis).
#
# Corridor total width = corridor_alpha * 3-D Euclidean distance between
# start_phys and goal_phys (in physical metres).
# Half-width d_max = 0.5 * corridor_alpha * ||goal_phys - start_phys||.
#
# If corridor_alpha < 0 the function delegates to optimize_path_casadi
# with no extra constraints.
# ---------------------------------------------------------------------------
def optimize_path_casadi_corridor(
    path_init,
    sdf_fun,
    wind_fun,
    clearance,
    v_max,
    a_max,
    t_max,
    dt0,
    map_shape,
    grid_scale=(4.0, 1.5, 4.0),
    dt_min=1e-3,
    dt_max=500.0,
    seg_samples=2,
    max_iter=200,
    verbose=True,
    corridor_alpha: float = -1.0,
):
    """
    Optimize path energy with CasADi, with an optional horizontal corridor
    constraint.

    All parameters are identical to optimize_path_casadi, plus:

    corridor_alpha : float
        If < 0 → no corridor (delegates to optimize_path_casadi unchanged).
        Otherwise, every optimised waypoint (and every intra-segment sample
        point) must satisfy:
            lateral distance to start-goal line  <=  d_max
        where d_max = 0.5 * corridor_alpha * ||goal_phys - start_phys||_2.
        The constraint is expressed in squared form to keep it smooth and
        compatible with IPOPT:
            d_max^2 * ||d||^2 - cross^2  >= 0
        where d = (G_x-S_x, G_y-S_y) and
              cross = (p_x-S_x)*d_y - (p_y-S_y)*d_x
        with the horizontal axes being physical coordinates 0 and 2.

    Physical coordinate layout (consistent with build_interpolants):
        axis 0 → X  (grid_scale[0], horizontal)
        axis 1 → Z  (grid_scale[1], height)
        axis 2 → Y  (grid_scale[2], horizontal)
    """
    # --- No corridor: plain optimizer ---
    if corridor_alpha < 0.0:
        return optimize_path_casadi(
            path_init,
            sdf_fun,
            wind_fun,
            clearance,
            v_max,
            a_max,
            t_max,
            dt0,
            map_shape,
            grid_scale=grid_scale,
            dt_min=dt_min,
            dt_max=dt_max,
            seg_samples=seg_samples,
            max_iter=max_iter,
            verbose=verbose,
        )

    # Same NLP as optimize_path_casadi, plus the corridor constraints marked below.
    grid_scale = np.asarray(grid_scale, dtype=float)
    map_shape  = np.asarray(map_shape,  dtype=float)
    path_init  = np.asarray(path_init,  dtype=float)
    path_phys  = path_init * grid_scale
    n          = path_phys.shape[0]
    start_phys = path_phys[0]
    goal_phys  = path_phys[-1]

    # --- Pre-compute corridor parameters (numpy scalars / floats) ---
    # Horizontal plane: physical axes 0 and 2.
    S_x = float(start_phys[0])
    S_y = float(start_phys[2])
    G_x = float(goal_phys[0])
    G_y = float(goal_phys[2])
    d_x  = G_x - S_x
    d_y  = G_y - S_y
    d_norm_sq = d_x * d_x + d_y * d_y      # ||d||^2  (m^2)

    dist_3d  = float(np.linalg.norm(goal_phys - start_phys))
    d_max    = 0.5 * corridor_alpha * dist_3d
    d_max_sq = d_max * d_max

    # Corridor is skipped when start == goal horizontally (degenerate).
    use_corridor = d_norm_sq > 1e-12

    x  = ca.MX.sym("x", n, 3)
    v  = ca.MX.sym("v", n, 3)
    a  = ca.MX.sym("a", n, 3)
    dt = ca.MX.sym("dt")

    # --- Physical model parameters (identical to optimize_path_casadi) ---
    rho        = 1.18
    Cd         = 1.0
    Aframe     = 0.16
    m          = 5.0
    A          = 0.5
    P_p_hover  = 100.0
    g          = 9.81
    g_vec      = ca.DM([0.0, -g, 0.0])
    eps        = 1e-9

    g_list = []
    lbg    = []
    ubg    = []
    power  = []

    # --- Kinematic constraints (trapezoidal integration) ---
    for i in range(n - 1):
        xi   = x[i,   :].T
        xip1 = x[i+1, :].T
        vi   = v[i,   :].T
        vip1 = v[i+1, :].T
        ai   = a[i,   :].T
        aip1 = a[i+1, :].T
        x_next = xi + 0.5 * (vi + vip1) * dt
        v_next = vi + 0.5 * (ai + aip1) * dt
        g_list.append(xip1 - x_next)
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]
        g_list.append(vip1 - v_next)
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]

    # --- Boundary position constraints ---
    g_list.append(x[0,  :].T - start_phys)
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]
    g_list.append(x[-1, :].T - goal_phys)
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]

    # --- Pointwise constraints and power terms ---
    for i in range(n):
        xi = x[i, :].T
        vi = v[i, :].T
        ai = a[i, :].T

        # Collision clearance
        g_list.append(sdf_fun(xi) - clearance)
        lbg.append(0.0)
        ubg.append(ca.inf)

        # --- CORRIDOR CONSTRAINT (node) ---
        # Constraint: d_max^2 * ||d||^2 - cross^2 >= 0
        # where cross = (xi_x - S_x)*d_y - (xi_y_horiz - S_y)*d_x
        # xi[0] = physical X (horizontal), xi[2] = physical Y (horizontal)
        if use_corridor:
            cross_i = (xi[0] - S_x) * d_y - (xi[2] - S_y) * d_x
            g_list.append(d_max_sq * d_norm_sq - cross_i * cross_i)
            lbg.append(0.0)
            ubg.append(ca.inf)

        # Acceleration limit
        g_list.append(a_max - _smooth_norm(ai))
        lbg.append(0.0)
        ubg.append(ca.inf)

        # Aerodynamics for thrust constraint and energy
        if wind_fun is None:
            v_w = ca.MX.zeros(3, 1)
        else:
            v_w = wind_fun(xi)
        v_a = vi - v_w

        # Airspeed limit
        g_list.append(v_max - _smooth_norm(v_a))
        lbg.append(0.0)
        ubg.append(ca.inf)

        v_a_norm = ca.sqrt(ca.dot(v_a, v_a) + eps)
        D        = -0.5 * rho * Cd * Aframe * (v_a_norm * v_a)

        T_vec  = m * ai - m * g_vec - D
        T_norm = _smooth_norm(T_vec)

        # Thrust limit
        g_list.append(t_max - T_norm)
        lbg.append(0.0)
        ubg.append(ca.inf)

        t_hat  = T_vec / (T_norm + eps)
        v_c    = ca.dot(v_a, t_hat)
        v_i    = -0.5 * v_c + ca.sqrt((0.5 * v_c) ** 2 + T_norm / (2.0 * rho * A) + eps)

        P_useful  = ca.dot(T_vec, v_a)
        P_induced = T_norm * v_i
        P_p       = P_p_hover * (T_norm / (m * g)) ** 1.5
        power.append(P_useful + P_induced + P_p)

    # --- Segment interior collision checks (linear interpolation) ---
    if seg_samples > 0:
        for i in range(n - 1):
            xi   = x[i,   :].T
            xip1 = x[i+1, :].T
            for k in range(1, seg_samples + 1):
                alpha_seg = float(k) / float(seg_samples + 1)
                x_seg     = (1.0 - alpha_seg) * xi + alpha_seg * xip1

                # Collision clearance on segment interior point
                g_list.append(sdf_fun(x_seg) - clearance)
                lbg.append(0.0)
                ubg.append(ca.inf)

                # --- CORRIDOR CONSTRAINT (segment interior point) ---
                if use_corridor:
                    cross_seg = (x_seg[0] - S_x) * d_y - (x_seg[2] - S_y) * d_x
                    g_list.append(d_max_sq * d_norm_sq - cross_seg * cross_seg)
                    lbg.append(0.0)
                    ubg.append(ca.inf)

    # --- Total energy (trapezoidal integration) ---
    E_total = 0
    for i in range(n - 1):
        E_total += 0.5 * (power[i] + power[i + 1]) * dt

    # --- Decision vector ---
    z = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(v, -1, 1), ca.reshape(a, -1, 1), dt)
    g = ca.vertcat(*g_list)

    # --- Initial guess ---
    v0, a0, dt0_eff = _build_feasible_initial_guess(path_phys, dt0, v_max, a_max)
    z0 = np.concatenate(
        [
            path_phys.reshape(-1, order="F"),
            v0.reshape(-1, order="F"),
            a0.reshape(-1, order="F"),
            [dt0_eff],
        ]
    )

    # --- Variable bounds ---
    lbx = -ca.inf * np.ones(z.shape[0])
    ubx =  ca.inf * np.ones(z.shape[0])
    map_upper = (map_shape - 1.0) * grid_scale
    x_lower   = np.zeros(n * 3)
    x_upper   = np.concatenate(
        [
            np.full(n, map_upper[0]),
            np.full(n, map_upper[1]),
            np.full(n, map_upper[2]),
        ]
    )
    lbx[: n * 3] = x_lower
    ubx[: n * 3] = x_upper
    lbx[-1] = dt_min
    ubx[-1] = dt_max

    nlp  = {"x": z, "f": E_total, "g": g}
    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "print_time":        1 if verbose else 0,
        "ipopt.max_iter":    int(max_iter),
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    try:
        sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    except RuntimeError as exc:
        return None, None, {"success": False, "message": str(exc)}

    z_opt  = sol["x"].full().squeeze()
    x_opt  = z_opt[: n * 3].reshape(n, 3, order="F")
    v_opt  = z_opt[n * 3 : n * 6].reshape(n, 3, order="F")
    a_opt  = z_opt[n * 6 : n * 9].reshape(n, 3, order="F")
    dt_opt = float(z_opt[-1])

    path_opt = x_opt / grid_scale
    info = {
        "success":    True,
        "status":     solver.stats()["return_status"],
        "iterations": solver.stats().get("iter_count", None),
        "dt":         dt_opt,
        "v_opt":      v_opt,
        "a_opt":      a_opt,
        "obj_f":      float(sol["f"]),
    }

    return path_opt, float(sol["f"]), info
