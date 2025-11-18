# BeaVR
**Bimanual, multi-Embodiment, Accessible VR Teleoperation for Robots**

<p align="center">
  <img src="media/BeaVR_logo.svg" alt="BeaVR-Bot Logo" width="300"/>
</p>

<p align="center">
  <strong>Alejandro Posadas-Nava</strong> ·
  <strong>Alejandro Carrasco</strong> ·
  <strong>Richard Linares</strong>
</p>

<p align="center">
  <br>
  <a href="https://arxiv.org/abs/2508.09606">
    <img src="https://yuxiaoba.github.io/assets/images/badges/Arxiv.png" alt="arXiv" width="14" style="vertical-align:middle;"/> Paper
  </a> |
  <a href="https://arclab-mit.github.io/beavr-landing/">
    <img src="https://images.icon-icons.com/3685/PNG/512/github_logo_icon_229278.png" alt="arXiv" width="14" style="vertical-align:middle;"/> Project Page
  </a> |
  <a href="https://github.com/ARCLab-MIT/BeaVR-app">
    <img src="https://images.icon-icons.com/3053/PNG/512/unity_hub_macos_bigsur_icon_189587.png" alt="arXiv" width="16" style="vertical-align:middle;"/> VR App
  </a>
</p>

---

## Overview

BeaVR is an open-source, end-to-end teleoperation pipeline that leverages affordable hardware for robotic teleoperation.

Key features:
- **VR teleoperation out-of-the-box** – Stream low-latency control and visual feedback through Meta Quest 3 (and any OculusVR supported device) while recording synchronized proprioceptive, visual, and action data.
- **Multi-embodiment support** – Ships with drivers and URDF assets for the RX-1 full-size humanoid and an xArm + LeapHand dexterous work-cell. The modular hardware abstraction layer lets you drop-in new robots with a single interface file.
- **Simulation parity** – Mirror every real-world session in MuJoCo or Isaac Gym for rapid domain-randomized policy training or sim-to-real transfer.
- **Dexterous demonstration collection** – Capture single-hand, bi-manual, or whole-body demonstrations for manipulation, assembly, or locomotion tasks—no motion-capture stage required.
- **Budget-friendly extensibility** – Works on commodity PCs and laptops with accessible robotics hardware.

### Why use BeaVR?
- **Accessible** – No proprietary hardware or licenses; every component is student-budget friendly and permissively BSD-3-licensed.
- **Modular & maintainable** – Clean and performant python and ROS modules.
- **LeRobot formatted data** – Standarizing data collection for shared robotics projects.
- **Community-driven** – Contributions already include UR-series arms, quadrupeds, and tactile grippers; PRs with new morphologies are welcome.

## Quick Start

### Prerequisites
- Linux system or containerized environment (see [Docker folder](docker))
- NVIDIA GPU with CUDA support (recommended)
- Meta Quest 3 VR headset
- [uv](https://github.com/astral-sh/uv) package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Build tools: `build-essential`, `python3-dev` (for compiling C extensions)
- Rust compiler (optional, only needed if building from source): Install via [rustup](https://rustup.rs/) or `sudo apt install rustup && rustup default stable`

### Installation

#### uv Environment (Recommended)

For development and running the full system with all dependencies (including PyTorch and simulation tools), we recommend using uv:

```bash
# Install Python 3.10.13
uv python install 3.10.13

# Create virtual environment with Python 3.10.13
uv venv --python 3.10.13

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

After setting up the environment, install BeaVR as a package using uv:

```bash
# Install all dependencies including dev extras
uv sync --extra dev

# This will create a uv.lock file and install everything
```

Alternatively, you can use pip:
```bash
pip install -e .[dev]
```

Setup pre-commit hooks:
```bash
pre-commit install
```

Verify the installation:
```bash
python -c "import sys;
try: import beavr; print('BeaVR successfully installed!')
except ImportError: print('An error occurred'); sys.exit(1)"
```

## Documentation

Full documentation lives in the [`docs`](docs) directory. Start with
[`docs/README.md`](docs/README.md) for an overview of the available guides,
including detailed explanations of the teleoperation and LeRobot stacks.

## Additional Features

### Apple Vision Pro Support
BeaVR only requires cartesian positions from VR headsets in the standard y-up right-hand coordinate frame used by VR systems.
Apple Vision Pro users can connect through the third-party Improbable AI's [Tracking Streamer App](https://github.com/Improbable-AI/VisionProTeleop) that provides independent hand pose tracking.
This app should seamlessly integrate with BeaVR as an alternative VR endpoint. Although, it has not been developed or tested for this purpose.

## Citation

If you use BeaVR in your research, please cite our work:

```bibtex
@misc{posadasnava2025beavr,
  title         = {BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots},
  author        = {Alejandro Posadas-Nava and Alejandro Carrasco and Richard Linares},
  year          = {2025},
  eprint        = {2508.09606},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Accepted for presentation at ICCR 2025, Kyoto},
  url           = {https://arxiv.org/abs/2508.09606}
}
```

## License

# MIT License

Copyright (c) 2025 MIT Arclab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.


## Acknowledgments

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/US_Air_Force_Logo_Solid_Colour.svg" alt="USAF Logo" width="60"/>
</p>

<p align="justify">
This work was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
</p>

<p align="center">
<sub><sup>© 2025 Massachusetts Institute of Technology</sup></sub>
</p>
