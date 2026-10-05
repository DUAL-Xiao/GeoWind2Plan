"""Plan UAV missions in one urban block under the four planning winds.

Examples
--------
    # one mission, printed to the terminal
    python run_planning_experiment.py --block A --mission 0

    # the full experiment for block A at 4 m/s, 75 m (256 missions)
    python run_planning_experiment.py --block A --workers 16

    # wind speed 8 m/s, start/goal altitude 100 m, 20 % corridor
    python run_planning_experiment.py --block A --wind-speed 8
    python run_planning_experiment.py --block A --altitude 100
    python run_planning_experiment.py --block A --corridor-width 0.2

Every selected trajectory is evaluated under the CFD reference wind; the
result file is read by summarize_results.py.
"""
import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from geowind2plan.data_loading import METHODS, METHOD_NAMES, load_block
from geowind2plan.mission_planner import SETTINGS, Planner
from geowind2plan.mission_sampling import sample_start_goal_inset_border

SEED = 0
MISSION_SAMPLING = dict(inset=5, clearance=SETTINGS["clearance_m"], min_chord_ratio=0.9)

_PLANNER = None     # one per worker process


def _init_worker(block_data, corridor_width):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    global _PLANNER
    _PLANNER = Planner(block_data, corridor_width)


def _solve(job):
    mission, start, goal = job
    return _PLANNER.plan(mission, start, goal, seed=SEED + mission)


def altitude_layer(altitude_m, spacing, z_offset, n_layers):
    """Grid layer of the start and goal: the first layer at or above the altitude."""
    layer = math.ceil(altitude_m / spacing[1] - 1e-9) - round(z_offset / spacing[1])
    if not 0 <= layer < n_layers:
        raise ValueError(f"altitude {altitude_m} m is outside the "
                         f"{z_offset:g}-{z_offset + (n_layers - 1) * spacing[1]:g} m flight band")
    return layer


def sample_missions(sdf, layer, count):
    rng = np.random.default_rng(SEED)
    return [(i, *sample_start_goal_inset_border(sdf, z0=layer, rng=rng, **MISSION_SAMPLING))
            for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", required=True, choices=list("ABCDE"))
    parser.add_argument("--wind-speed", type=float, default=4.0, help="inflow speed in m/s (default 4)")
    parser.add_argument("--altitude", type=float, default=75.0, help="start/goal altitude in m (default 75)")
    parser.add_argument("--corridor-width", type=float, default=None,
                        help="corridor width as a fraction of the start-goal distance (default: none)")
    parser.add_argument("--missions", type=int, default=256, help="number of missions (default 256)")
    parser.add_argument("--mission", type=int, default=None, help="plan only this mission and print it")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--data-dir", default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out", default=None, help="result file (default: results/<settings>.npy)")
    args = parser.parse_args()

    block = load_block(args.data_dir, args.block, args.wind_speed)
    layer = altitude_layer(args.altitude, block["spacing"], block["z_offset"], block["sdf"].shape[1])

    if args.mission is not None:
        (job,) = sample_missions(block["sdf"], layer, args.mission + 1)[-1:]
        _init_worker(block, args.corridor_width)
        result = _solve(job)
        distance = np.linalg.norm(result["goal_m"] - result["start_m"])
        print(f"mission {job[0]}: {result['start_m'].round(1)} -> {result['goal_m'].round(1)} m, "
              f"straight-line distance {distance:.0f} m")
        for method in METHODS:
            r = result["methods"][method]
            if r is None:
                print(f"  {METHOD_NAMES[method]:14} no feasible trajectory")
                continue
            print(f"  {METHOD_NAMES[method]:14} {r['energy_cfd_J'] / distance / 3.6:6.2f} Wh/km "
                  f"(evaluated in CFD wind), mean altitude {r['path_m'][:, 1].mean():5.1f} m")
        return

    # In a narrow corridor some start-goal pairs have no route, or only an
    # extremely narrow one, inside the flight envelope: a building rising above
    # the envelope can span the whole corridor.  Twice as many pairs are drawn,
    # and those for which a feasible trajectory can hardly be found are dropped.
    n_draw = args.missions * (2 if args.corridor_width is not None else 1)
    jobs = sample_missions(block["sdf"], layer, n_draw)

    t0 = time.time()
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(block, args.corridor_width)) as pool:
        missions = []
        for result in pool.imap_unordered(_solve, jobs, chunksize=1):
            missions.append(result)
            print(f"[{len(missions)}/{n_draw}] mission {result['mission']} "
                  f"elapsed {time.time() - t0:7.0f} s", flush=True)

    if args.corridor_width is not None:
        missions.sort(key=lambda m: (-m["n_feasible_trials"], m["mission"]))
        missions = missions[:args.missions]
    missions.sort(key=lambda m: m["mission"])

    if args.out is None:
        name = f"block_{args.block}_wind_{args.wind_speed:g}ms_altitude_{args.altitude:g}m"
        if args.corridor_width is not None:
            name += f"_corridor_{args.corridor_width:g}"
        args.out = Path("results") / f"{name}.npy"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    settings = dict(SETTINGS, block=args.block, wind_speed_m_s=args.wind_speed,
                    altitude_m=args.altitude, corridor_width=args.corridor_width,
                    seed=SEED, **MISSION_SAMPLING)
    np.save(args.out, {"settings": settings, "missions": missions}, allow_pickle=True)
    print(f"saved {len(missions)} missions to {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
