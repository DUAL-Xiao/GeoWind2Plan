import numpy as np


def _sample_wind_fun(positions, wind_fun):
    """Sample wind from an interpolant function at physical positions."""
    v_w = np.zeros_like(positions, dtype=float)
    for i, p in enumerate(positions):
        w = wind_fun(p)
        if hasattr(w, "full"):
            w = w.full()
        w = np.asarray(w, dtype=float).reshape(-1)
        v_w[i] = w
    return v_w


def get_energy_xv(positions, velocities, wind_fun=None, eps=1e-9):
    """
    Compute total energy using positions and velocities (physical units).

    Args:
        positions (np.ndarray): shape (n, 3), physical coordinates (meters).
        velocities (np.ndarray): shape (n, 3), ground velocities (m/s).
        wind_fun (callable|None): optional wind interpolant in physical coords.
        eps (float): numerical epsilon.

    Returns:
        tuple: (E_total, E_useful, E_induced, E_p)
    """
    P = np.asarray(positions, dtype=float)
    V = np.asarray(velocities, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    dP = P[1:] - P[:-1]
    ds = np.linalg.norm(dP, axis=1)
    v_norm = np.linalg.norm(V, axis=1)
    v_avg = 0.5 * (v_norm[:-1] + v_norm[1:])
    dt = ds / np.where(v_avg > eps, v_avg, eps)

    t = np.zeros(n)
    t[1:] = np.cumsum(dt)

    # Acceleration from velocity using central differences in time.
    a = np.zeros_like(V)
    if n > 2:
        dt1 = t[1:-1] - t[:-2]
        dt2 = t[2:] - t[1:-1]
        dv1 = V[1:-1] - V[:-2]
        dv2 = V[2:] - V[1:-1]
        a[1:-1] = (dv1 + dv2) / (dt1 + dt2)[:, None]
        a[0] = a[1]
        a[-1] = a[-2]
    else:
        a[0] = (V[1] - V[0]) / max(dt[0], eps)
        a[1] = a[0]

    # Wind sampling via interpolant (physical coordinates)
    if wind_fun is None:
        v_w = np.zeros_like(V)
    else:
        v_w = _sample_wind_fun(P, wind_fun)

    v_a = V - v_w

    # Energy model parameters
    rho = 1.18
    Cd = 1.0
    Aframe = 0.16
    m = 5.0
    A = 0.5
    P_p_hover = 100.0
    g = 9.81
    g_vec = np.array([0.0, -g, 0.0])

    v_a_norm = np.sqrt(np.sum(v_a**2, axis=1) + eps)
    D = -0.5 * rho * Cd * Aframe * (v_a_norm[:, None] * v_a)

    T_vec = m * a - m * g_vec - D
    T_norm = np.sqrt(np.sum(T_vec**2, axis=1) + eps)
    t_hat = T_vec / (T_norm[:, None] + eps)
    v_c = np.sum(v_a * t_hat, axis=1)
    v_i = -0.5 * v_c + np.sqrt((0.5 * v_c) ** 2 + T_norm / (2.0 * rho * A) + eps)

    P_useful = np.sum(T_vec * v_a, axis=1)
    P_induced = T_norm * v_i
    P_p = P_p_hover * (T_norm / (m * g)) ** 1.5

    def integrate(Pcomp):
        return np.sum(0.5 * (Pcomp[1:] + Pcomp[:-1]) * dt)

    E_useful = integrate(P_useful)
    E_induced = integrate(P_induced)
    E_p = integrate(P_p)
    E_total = E_useful + E_induced + E_p

    return E_total, E_useful, E_induced, E_p
