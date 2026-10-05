"""Loading an urban block and the wind fields used for planning and evaluation.

Array layout
------------
Every field is stored on the block's grid with axes (axis 0, axis 1, axis 2),
where axis 1 is vertical and axes 0 and 2 are horizontal.

Wind fields have shape (n0, n1, n2, 3); wind[..., k] is the velocity component
along array axis k, in m/s.  The background wind blows along +axis 2.

All fields were computed on the full computational domain and then cropped to
the 30-120 m flight band, so values at the edges of the band are those of the
full-domain fields.
"""
from pathlib import Path

import numpy as np

METHODS = ("cfd", "ours", "profile", "agnostic")
METHOD_NAMES = {
    "cfd": "Ground truth",
    "ours": "Ours",
    "profile": "Profile wind",
    "agnostic": "Wind-agnostic",
}


GRID_SPACING_M = (4.0, 1.5, 4.0)      # along axis 0, axis 1 (vertical), axis 2
LOWEST_LAYER_HEIGHT_M = 30.0          # height of axis-1 index 0 above ground
REFERENCE_WIND_SPEED = 4.0            # inflow speed (m/s) of wind_predicted and the profile wind


def load_block(data_dir, block, wind_speed):
    """Geometry and the four planning winds of one urban block.

    The predicted wind and the profile wind are given at the reference inflow
    speed and are scaled by wind_speed / reference speed.  The CFD reference
    wind is a separate simulation for each inflow speed.

    Returns a dict with keys
        spacing, z_offset, buildings, sdf, wind_cfd,
        and wind_<method> for the planning winds ("agnostic" is None).
    """
    data_dir = Path(data_dir)
    folder = data_dir / f"block_{block}"
    if not folder.is_dir():
        raise FileNotFoundError(f"no data for urban block {block!r} in {data_dir}")

    scale = float(wind_speed) / REFERENCE_WIND_SPEED
    if scale == 1.0:
        cfd_file = folder / "wind_cfd.npy"
    else:
        cfd_file = folder / f"wind_cfd_{wind_speed:g}ms.npy"
        if not cfd_file.exists():
            raise FileNotFoundError(
                f"block {block} has no CFD reference for {wind_speed:g} m/s ({cfd_file.name})")

    sdf = np.load(folder / "sdf.npy")
    wind_cfd = np.load(cfd_file)

    wind_ours = np.load(folder / "wind_predicted.npy")
    if scale != 1.0:
        wind_ours = wind_ours * scale

    # Height-only profile of the along-wind component; no horizontal structure.
    profile = np.load(data_dir / "profile_wind.npy")
    wind_profile = np.zeros(sdf.shape + (3,), dtype=profile.dtype)
    wind_profile[:, :, :, 2] = profile[None, :, None]
    if scale != 1.0:
        wind_profile = wind_profile * scale

    return {
        "spacing": np.asarray(GRID_SPACING_M, dtype=float),
        "z_offset": LOWEST_LAYER_HEIGHT_M,
        "buildings": np.load(folder / "buildings.npy"),
        "sdf": sdf,
        "wind_cfd": wind_cfd,
        "wind_ours": wind_ours,
        "wind_profile": wind_profile,
        "wind_agnostic": None,
    }
