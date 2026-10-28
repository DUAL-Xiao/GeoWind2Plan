# GeoWind2Plan

Code and data for the UAV planning experiments of *GeoWind2Plan: Mission-Time 3D
Urban Wind Prediction for Energy-Efficient UAV Planning*.

Demo: https://geowind2plan.com/

Each mission is planned with four planning winds: the CFD wind (Ground truth),
the neural-operator prediction (Ours), a height-only wind profile (Profile wind)
and zero wind (Wind-agnostic). The predicted wind fields for all experiments of
the paper are provided in `data/`.

## Installation

```
pip install -r requirements.txt
```

## Code

```
run_planning_experiment.py   runs the planning experiment for one urban block
summarize_results.py         prints energy and flight statistics of result files
geowind2plan/
  data_loading.py            loads the geometry and the four planning winds
  mission_sampling.py        samples start and goal positions
  mission_planner.py         plans one mission with each planning wind and selects the trajectory
  rrt_path_search.py         RRT search for collision-free initial paths
  trajectory_optimization.py optimizes position, velocity and timing with CasADi and IPOPT
  energy_model.py            UAV energy model used for selection and evaluation
```

## Data

Download the contents of the `geowind2plan_data/` folder of
[CityFFD-UrbanMicroclimate](https://huggingface.co/datasets/UMBE-LAB/CityFFD-UrbanMicroclimate)
and place them in `data/`:

```
data/
  profile_wind.npy      profile wind, one value per layer (m/s)
  block_A/ ... block_E/
    buildings.npy       building occupancy
    sdf.npy             signed distance to buildings (m), positive in free space
    wind_cfd.npy        CFD wind, 4 m/s inflow
    wind_predicted.npy  predicted wind, 4 m/s inflow
  block_A/
    wind_cfd_2ms.npy    CFD wind, 2 m/s inflow
    wind_cfd_8ms.npy    CFD wind, 8 m/s inflow
```

All fields were computed on the full domain and then cropped to the 30-120 m
flight band. The signed distance therefore also accounts for buildings outside
the band. The lowest stored layer is at 30 m.

Arrays have axes (axis 0, axis 1, axis 2), with axis 1 vertical. The grid
spacing is 4 m along axes 0 and 2 and 1.5 m along axis 1. Wind arrays
have shape `(n0, n1, n2, 3)`, and `wind[..., k]` is the velocity component
along axis `k`. The background wind blows along +axis 2.

The predicted wind and the profile wind are given for the 4 m/s reference
inflow. `run_planning_experiment.py` rescales them for other wind speeds. The
profile wind is the along-wind component and is applied uniformly over each
horizontal layer.

## Usage

### run_planning_experiment.py

Plans the missions of one urban block with the four planning winds and
evaluates every selected trajectory in the CFD wind.

Input: the data in `data/` and the settings `--block` (A to E), `--wind-speed`
(m/s, default 4), `--altitude` (m, default 75), `--corridor-width` (fraction of
the start-goal distance, default none) and `--missions` (default 256).

Output: `results/block_<block>_wind_<speed>ms_altitude_<altitude>m.npy`, with
`_corridor_<width>` appended when a corridor is used. For every mission it
holds the start and goal and, for each of the four methods, the selected
trajectory and its energy in the CFD wind.

```
python run_planning_experiment.py --block A --workers 16
```

Missions are solved in parallel. Set `--workers` to the number of processes
your machine can run at once.

With `--mission k`, only mission k is planned and its energies are printed to
the screen:

```
python run_planning_experiment.py --block A --mission 0
```

Trajectories must stay within the 30-120 m flight envelope, so in a narrow
corridor a building taller than 120 m can block every route between start and
goal. With `--corridor-width`, the script draws 512 start-goal pairs and
discards those for which a feasible route can hardly be found, leaving 256
missions.

### summarize_results.py

Input: one or more result files of `run_planning_experiment.py`.

Output: printed tables for all missions and for tailwind, headwind and
crosswind missions. For each method they give the unit-distance energy (Wh/km)
in the CFD wind as mean and standard deviation, the extra energy relative to
Ground truth in percent, and the mean flight altitude and mean ground speed.

```
python summarize_results.py results/block_A_wind_4ms_altitude_75m.npy
```

## Citation

```bibtex
@misc{qin2026geowind2planmissiontime3durban,
  title={GeoWind2Plan: Mission-Time 3D Urban Wind Prediction for Energy-Efficient UAV Planning},
  author={Shaoxiang Qin and Yucheng Zhao and Fuyuan Lyu and Di Zhou and Jiachen Yao and Xue Liu and Anima Anandkumar and Liangzhu Leon Wang and Xiongye Xiao},
  year={2026},
  eprint={2609.36056},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.36056},
}
```
