# Hierarchical Adaptive Network for Dexterous Soft manipulation (HANDS)
This repository implements a multi-agent solution for controlling tendon driven soft manipulators for the purposes of object manipulation. The goal of this project was to develop a hierarchical RL controller for low level control of individual tendon driven soft robots and a higher level controller for coordination of multiple "finger" in the goal of manipulating some object in the scene.

This solution uses the simulation environment [PyElastica](https://github.com/GazzolaLab/PyElastica) and a tendon driven control plugin for PyElastica, [TendonForces](https://github.com/gabotuzl/TendonForces).

# Installation Instructions

To use this repository, please follow the steps below

## Prerequisites

Create a new Conda environment with python 3.11

```bash
conda create -n ENV_NAME python=3.11 -y
conda activate ENV_NAME
```

Install PyElastica from source

```bash
git clone git@github.com:GazzolaLab/PyElastica.git
cd PyElastica
pip install -e .
```

Clone and install this repository

```bash
git clone git@github.com:RowanRama/HANDS.git
cd HANDS
pip install -e .
```

## Documentation

The HANDS subfolder contains TendonForces.py, which can be used as a forcing function
The examples folder has a visualize_forces.py, which can be run to simulate a basic tendon-driven soft robot

To run RL on a single finger tracing a sphere first run test_rl_env.py. The ouptput saves as test_sphere.pkl, which can be passed into gen_gif.py for a visualization.


Concept	Meaning
Episode:	A full run from env.reset() until done=True
env.step():	One RL step = 1 action applied over num_steps_per_update * sim steps
n_steps=2048:	PPO collects 2048 RL steps total (across many episodes if needed)
Training freq:	PPO trains after every 2048 collected steps

