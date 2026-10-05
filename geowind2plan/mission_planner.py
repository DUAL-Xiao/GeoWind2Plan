"""Energy-aware planning of one mission under the four planning winds.

For each mission the planner draws several RRT warm starts.  Each warm start is
refined by the continuous trajectory optimizer once per planning wind, so all
four methods start from exactly the same collision-free initial paths.  For
each method the planner then keeps the candidate with the lowest energy under
that method's own planning wind (paper, Eq. 38).  The CFD reference wind is
used for nothing but the final evaluation of the selected trajectory, except
for the "cfd" method, whose planning wind is the CFD wind by definition.
"""
import random

import numpy as np

from .data_loading import METHODS
from .energy_model import get_energy_xv
from .rrt_path_search import RRTPlanner3DCorridor
from .trajectory_optimization import build_interpolants, optimize_path_casadi_corridor

# Planner and vehicle settings used for every experiment in the paper.
SETTINGS = {
    "rrt_initializations": 10,
    "clearance_m": 10.0,
    "max_airspeed_m_s": 20.0,
    "max_acceleration_m_s2": 3.0,
    "max_thrust_N": 100.0,
    "initial_time_step_s": 1.0,
    "segment_samples": 2,
}

RRT_SETTINGS = dict(
    step_size=15,
    min_step_size=5,
    sdf_scale=0.4,
    max_iters=2000,
    goal_sample_rate=0.1,
    shortcut_iters=0,
    shortcut_max_span=3,
    smooth_iters=0,
    el_gain=0.15,
    rep_gain=30,
    repulsive_radius=50.0,
)


class Planner:
    """Everything that is expensive to build once per process."""

    def __init__(self, block_data, corridor_width=None):
        self.spacing = block_data["spacing"]
        self.z_offset = block_data["z_offset"]
        self.sdf = block_data["sdf"]
        # A negative width means no corridor constraint.
        self.corridor = -1.0 if corridor_width is None else float(corridor_width)

        self.rrt = RRTPlanner3DCorridor(
            block_data["buildings"], self.sdf,
            clearance=SETTINGS["clearance_m"], **RRT_SETTINGS)

        self.sdf_fun, cfd_fun = build_interpolants(self.sdf, block_data["wind_cfd"], self.spacing)
        self.wind_fun = {"cfd": cfd_fun, "agnostic": None}
        for method in ("ours", "profile"):
            _, self.wind_fun[method] = build_interpolants(
                self.sdf, block_data[f"wind_{method}"], self.spacing)
        self.cfd_fun = cfd_fun

    def candidates(self, start, goal, seed):
        """All optimized candidates of one mission, per method.

        start and goal are grid indices inside the flight band.  Returns
        (candidates, n_feasible_trials): candidates[method] is a list with one
        entry per successful optimization; n_feasible_trials counts the warm
        starts for which RRT and all four optimizations succeeded.
        """
        np.random.seed(seed)
        random.seed(seed)

        found = {m: [] for m in METHODS}
        n_feasible = 0
        for _ in range(SETTINGS["rrt_initializations"]):
            path = self.rrt.plan(start.tolist(), goal.tolist(), corridor_alpha=self.corridor)
            if path is None:
                continue
            path = np.asarray(path, dtype=float)

            all_ok = True
            for method in METHODS:
                opt_path, _, info = optimize_path_casadi_corridor(
                    path_init=path,
                    sdf_fun=self.sdf_fun,
                    wind_fun=self.wind_fun[method],
                    clearance=SETTINGS["clearance_m"],
                    v_max=SETTINGS["max_airspeed_m_s"],
                    a_max=SETTINGS["max_acceleration_m_s2"],
                    t_max=SETTINGS["max_thrust_N"],
                    dt0=SETTINGS["initial_time_step_s"],
                    map_shape=self.sdf.shape,
                    grid_scale=self.spacing,
                    seg_samples=SETTINGS["segment_samples"],
                    verbose=False,
                    corridor_alpha=self.corridor,
                )
                all_ok = all_ok and opt_path is not None and info.get("success", False)
                if opt_path is None:
                    continue
                position = (np.asarray(opt_path, dtype=float) * self.spacing).astype(np.float32)
                velocity = np.asarray(info["v_opt"])
                found[method].append({
                    # Heights in the output are above ground, not above the band.
                    "path_m": position + np.array([0.0, self.z_offset, 0.0], dtype=np.float32),
                    "velocity_m_s": velocity,
                    "energy_planning_J": float(get_energy_xv(
                        position, velocity, wind_fun=self.wind_fun[method])[0]),
                    "energy_cfd_J": float(get_energy_xv(
                        position, velocity, wind_fun=self.cfd_fun)[0]),
                })
            n_feasible += int(all_ok)
        return found, n_feasible

    def plan(self, mission, start, goal, seed):
        """Plan one mission and keep, per method, the lowest planning-energy candidate."""
        found, n_feasible = self.candidates(start, goal, seed)
        offset = np.array([0.0, self.z_offset, 0.0])
        result = {
            "mission": int(mission),
            "start_m": start * self.spacing + offset,
            "goal_m": goal * self.spacing + offset,
            "n_feasible_trials": n_feasible,
            "methods": {},
        }
        for method, items in found.items():
            if not items:
                result["methods"][method] = None
                continue
            best = min(items, key=lambda c: c["energy_planning_J"])
            result["methods"][method] = dict(best, n_candidates=len(items))
        return result
