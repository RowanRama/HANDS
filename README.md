# Soft Manipulator Control
This repository implements a multi-agent solution for controlling tendon driven soft manipulators for the purposes of object manipulation. The goal of this project was to develop a hierarchical RL controller for low level control of individual tendon driven soft robots and a higher level controller for coordination of multiple "finger" in the goal of manipulating some object in the scene.

This solution uses the simulation environment [PyElastica](https://github.com/GazzolaLab/PyElastica) and a tendon driven control plugin for PyElastica, [TendonForces](https://github.com/gabotuzl/TendonForces).

# Installation Instructions

To use this repository, please follow the steps below

## Prerequisites

```bash
pip install pyelastica
pip install torch
pip install gymnasium
```

For the tendon driven addition to pyelastica:

```bash
git clone https://github.com/gabotuzl/TendonForces.git
```

Navigate to the cloned directory:
```bash
cd TendonForces
```

Add to the python path:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

