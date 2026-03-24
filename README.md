# scClone2DR: Clone-level multi-modal prediction of tumour drug response

<p align="center">
  <img src="assets/logo.png" alt="scClone2DR logo" width="500"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python Version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-BSD--3--Clause-02B36C.svg" alt="License">
  </a>
  <a href="https://hub.docker.com/r/quentinduchemin/scclone2dr">
    <img src="https://img.shields.io/badge/docker-scclone2dr-2496ED?logo=docker&logoColor=white" alt="Docker scclone2dr">
  </a>
  <a href="https://hub.docker.com/r/quentinduchemin/scclone2dr_ssh">
    <img src="https://img.shields.io/badge/docker-scclone2dr__ssh-2496ED?logo=docker&logoColor=white" alt="Docker scclone2dr_ssh">
  </a>
</p>

---

## Overview

**scClone2DR** is a probabilistic multi-modal framework for predicting drug responses at the level of individual tumour clones by integrating:

- **single-cell RNA sequencing (scRNA-seq)**
- **single-cell DNA sequencing (scDNA-seq)**
- **ex-vivo drug-screening data**


---

## Features

- **Multi-modal integration** of scRNA-seq, scDNA-seq, and drug-screening data
- **Probabilistic modelling** using [Pyro](https://pyro.ai/) for Bayesian inference
- **Clone-level drug response prediction** in heterogeneous tumour populations
- **Flexible training pipeline** for real and simulated datasets
- **Visualization and inference utilities** for downstream analysis

---

## Installation

You can install **scClone2DR** in three ways. We **recommend using the VS Code Dev Container** for the easiest and most reproducible workflow.

---

### Option 1 — Use VS Code Dev Container (recommended)

The easiest way to use **scClone2DR** is via a **VS Code Dev Container**:

1. Make sure you have:
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Docker](https://www.docker.com/) running
   - VS Code **Dev Containers** extension installed

2. Open your project folder (the folder containing `.devcontainer/` and `notebooks/`) in VS Code.

3. Press **Ctrl+Shift+P** (or Cmd+Shift+P on macOS) → search **Dev Containers: Reopen in Container** → press Enter.

4. VS Code will:
   - Pull the `quentinduchemin/scclone2dr` Docker image if necessary  
   - Mount your project folder into the container at `/workspace`  
   - Open a fully configured environment with Python, Jupyter, and all dependencies ready  

5. You can now open notebooks in `notebooks/` directly inside VS Code and run scClone2DR without additional setup.


### Option 2 — Install from source

```bash
git clone https://github.com/cbg-ethz/scClone2DR
cd scClone2DR
pip install -e .
pip install -e .[notebook]
```

### Option 3 — Use Docker without VSCode

Pre-built Docker images are available on Docker Hub:

- [`quentinduchemin/scclone2dr`](https://hub.docker.com/r/quentinduchemin/scclone2dr)  
  Standard runtime image (**without SSH service**)

- [`quentinduchemin/scclone2dr_ssh`](https://hub.docker.com/r/quentinduchemin/scclone2dr_ssh)  
  SSH-enabled runtime image (**for remote/containerized workflows**)

Pull the images:

```bash
docker pull quentinduchemin/scclone2dr
docker pull quentinduchemin/scclone2dr_ssh
```

Run the standard image:

```bash
docker run --rm -it \
  -v $(pwd):/workspace \
  -w /workspace \
  quentinduchemin/scclone2dr
```

Run the SSH-enabled image (example exposing port `2222`):

```bash
docker run --rm -it \
  -p 2222:22 \
  -v $(pwd):/workspace \
  -w /workspace \
  quentinduchemin/scclone2dr_ssh
```

---

## Docker Usage

### Build images locally (optional)

If you want to build the images from the repository instead of pulling them from Docker Hub:

```bash
docker build -t quentinduchemin/scclone2dr -f Dockerfile .
docker build -t quentinduchemin/scclone2dr_ssh -f Dockerfile.ssh .
```

---

## Quick Start

If you are using Docker, first start a container and run the following commands **inside** it.

```python
from scclone2dr.data import RealData
from scclone2dr.pipeline import scClone2DRPipeline
from scclone2dr.trainer import Trainer, GuideType

data_source = RealData(
    path_fastdrug="/path/to/FD_data.csv",
    path_rna="/path/to/rna_folder/",
)

data = data_source.get_real_data(
    concentration_DMSO=5,
    concentration_drug=5,
)

pipeline = scClone2DRPipeline(
    data_source=data_source,
    trainer=Trainer(guide_type=GuideType.FULL_MVN),
    mode_nu="noise_correction",
    mode_theta="not shared decoupled",
)

# Configure model topology from data source metadata
pipeline.model.configure(data_source)

params = pipeline.fit(
    data=data,
    n_steps=600,
    penalty_l1=0.1,
    penalty_l2=0.1,
)

pipeline.save("checkpoints/real_data_run.npz")
```

---

## Package Structure

```text
scClone2DR/
├── src/
│   └── scclone2dr/
│       ├── data/           # Real/simulated data loaders and dataset utilities
│       ├── baselines/      # FM / NN baseline models
│       ├── inference/      # Posterior sampling and model evaluation
│       ├── plots/          # Visualization helpers
│       ├── model.py        # Core probabilistic model definition
│       ├── trainer.py      # SVI training engine
│       ├── pipeline.py     # End-to-end orchestration API
│       ├── types.py        # Shared typing helpers
│       └── utils.py        # Utility functions
├── notebooks/
├── assets/
├── Dockerfile
├── Dockerfile.ssh
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Main API Modules

- `scclone2dr.data`  
  Data modules (`RealData`, `SimulatedData`, `BaseDataset`)

- `scclone2dr.model`  
  Core generative model (`scClone2DR`)

- `scclone2dr.trainer`  
  Training engine (`Trainer`, `GuideType`)

- `scclone2dr.pipeline`  
  High-level workflow (`scClone2DRPipeline`)

- `scclone2dr.inference`  
  Posterior sampling and evaluation utilities

- `scclone2dr.plots`  
  Plotting and visualization functions

---

## Requirements

### Core dependencies

- Python >= 3.8
- PyTorch >= 1.10.0
- Pyro >= 1.8.0
- NumPy >= 1.20.0
- pandas >= 1.3.0
- h5py >= 3.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.60.0
- scikit-learn >= 0.24.0
- scikit-fda >= 0.8.1

### Optional / notebook dependencies

- nbformat >= 5.0.0
- plotly >= 5.0.0

---

## Tutorial

For tutorials and examples:  [Tutorial Notebook](notebooks/tutorial_scClone2DR.ipynb)


---

## Citation

If you use **scClone2DR** in your research, please cite:

```bibtex
@article{scClone2DR2026,
  title={Clone-level multi-modal prediction of tumour drug response},
  author={Quentin Duchemin and Daniel Trejo Banos and Anne Bertolini and Pedro F. Ferreira and Rudolf Schill and Matthias Lienhard and Rebekka Wegmann and Tumor Profiler Consortium and Berend Snijder and Daniel Stekhoven and Niko Beerenwinkel and Franziska Singer and Guillaume Obozinski and Jack Kuipers},
  year={2026}
}
```

---

## License

This project is licensed under the **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for details.

---

## Contact

**Quentin Duchemin**  
quentin.duchemin@epfl.ch

**Project repository:**  
[https://github.com/cbg-ethz/scClone2DR](https://github.com/cbg-ethz/scClone2DR)
