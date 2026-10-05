"""Print energy and flight statistics for result files of run_planning_experiment.py.

    python summarize_results.py results/block_A_wind_4ms_altitude_75m.npy [more files ...]

For each file this prints, for all missions and for tailwind, headwind and
crosswind missions:
  * unit-distance energy (Wh/km) of each method, evaluated in the CFD wind,
    as mean +- standard deviation, with the extra energy relative to Ground truth;
  * mean flight altitude and mean ground speed.
"""
import argparse

import numpy as np

from geowind2plan.data_loading import METHODS, METHOD_NAMES

SECTORS = ("All", "Tailwind", "Headwind", "Crosswind")


def relative_wind_angle(start, goal):
    """Angle in degrees between the start-to-goal direction and the wind (+axis 2)."""
    d = np.asarray(goal, float) - np.asarray(start, float)
    return float(np.degrees(np.mod(np.arctan2(d[2], d[0]) - 0.5 * np.pi, 2.0 * np.pi)))


def sector_masks(missions):
    psi = np.array([relative_wind_angle(m["start_m"], m["goal_m"]) for m in missions])
    tail = (psi <= 45.0) | (psi >= 315.0)
    head = (psi >= 135.0) & (psi <= 225.0)
    return dict(zip(SECTORS, (np.ones_like(tail), tail, head, ~(tail | head))))


def per_mission(missions, method, quantity):
    values = np.full(len(missions), np.nan)
    for i, m in enumerate(missions):
        r = m["methods"][method]
        if r is None:
            continue
        if quantity == "energy":
            distance_km = np.linalg.norm(np.asarray(m["goal_m"]) - np.asarray(m["start_m"])) / 1000.0
            values[i] = r["energy_cfd_J"] / 3600.0 / distance_km
        elif quantity == "altitude":
            values[i] = np.asarray(r["path_m"])[:, 1].mean()
        else:
            values[i] = np.linalg.norm(np.asarray(r["velocity_m_s"]), axis=1).mean()
    return values


def stats(values, mask):
    v = values[mask & np.isfinite(values)]
    if v.size < 2:   # a sector can be empty in a small test run
        return (v.mean() if v.size else np.nan), np.nan
    return v.mean(), v.std(ddof=1)


def table(path):
    data = np.load(path, allow_pickle=True).item()
    missions, s = data["missions"], data["settings"]
    masks = sector_masks(missions)
    title = (f"Urban block {s['block']}, U = {s['wind_speed_m_s']:g} m/s, altitude {s['altitude_m']:g} m"
             + (f", corridor width {s['corridor_width']:g}" if s["corridor_width"] is not None else "")
             + f"  ({len(missions)} missions)")
    print(f"\n{title}\n" + "-" * len(title))

    energy = {m: per_mission(missions, m, "energy") for m in METHODS}
    print("Unit-distance energy (Wh/km)")
    print(f"{'':15}" + "".join(f"{name:>24}" for name in SECTORS))
    for method in METHODS:
        cells = []
        for sector in SECTORS:
            mean, sd = stats(energy[method], masks[sector])
            cell = f"{mean:.1f} +- {sd:.1f}"
            if method != "cfd":
                reference, _ = stats(energy["cfd"], masks[sector])
                cell += f" ({100.0 * (mean / reference - 1.0):+.1f}%)"
            cells.append(f"{cell:>24}")
        print(f"{METHOD_NAMES[method]:15}" + "".join(cells))

    for quantity, label, digits in (("altitude", "Mean altitude (m)", 1), ("speed", "Mean ground speed (m/s)", 2)):
        values = {m: per_mission(missions, m, quantity) for m in METHODS}
        print(label)
        print(f"{'':15}" + "".join(f"{METHOD_NAMES[m]:>15}" for m in METHODS))
        for sector in SECTORS:
            print(f"{sector:15}" + "".join(f"{stats(values[m], masks[sector])[0]:>15.{digits}f}" for m in METHODS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+")
    for path in parser.parse_args().results:
        table(path)


if __name__ == "__main__":
    main()
