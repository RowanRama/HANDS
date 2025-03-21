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


