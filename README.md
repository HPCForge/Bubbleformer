# Bubbleformer

**Accepted to NeurIPS 2025 as a Spotlight Paper🏅**

[[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/121854) [[Poster]](media/Poster_NIPS25_Bubbleformer_FINAL.pdf) [[Dataset on Hugging Face (BubbleML 2.0)]](https://huggingface.co/datasets/hpcforge/BubbleML_2)

A transformer model for long range forecasting of boiling phenomena (an inherently chaotic, multiphase process central to energy and thermal systems) that generalizes across different liquids such as dielectrics and refrigerants.

![Bubbleformer Overview](media/paper_overview.png)
*Figure 1: Overview of BubbleML 2.0 dataset and Bubbleformer downstream tasks*

![Task Overview](media/task_overview.png)
*Figure 2: Task overview showing the prediction and forecasting problem setup*

## Prior Work

This work builds upon **BubbleML**, which was accepted as a **NeurIPS 2023 Datasets and Benchmarks Spotlight**. BubbleML introduced the first large-scale multiphase boiling dataset for machine learning research.

- **BubbleML Paper**: [NeurIPS 2023 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/01726ae05d72ddba3ac784a5944fa1ef-Abstract-Datasets_and_Benchmarks.html)
- **BubbleML 1.0 & 2.0 Dataset**: [Hugging Face](https://huggingface.co/datasets/hpcforge/BubbleML_2)

## Overview


Bubbleformer is a transformer-based spatiotemporal model that forecasts stable and long-range boiling dynamics (including nucleation, liquid-vapor interface evolution, and heat transfer) without dependence on simulation data during inference. The project combines:


1. **Bubbleformer**: A novel transformer architecture for forecasting multiphase fluid dynamics
2. **BubbleML 2.0**: A comprehensive dataset of boiling simulations across diverse fluids and configurations

Together, they enable machine learning models to generalize across different fluids and physical configurations, setting new benchmarks for ML-based modeling of complex thermophysical systems.

## Bubbleformer Model

Bubbleformer makes three core contributions to the field:

1. **Beyond prediction to forecasting**
   - Operates directly on full 5D spatiotemporal tensors while preserving temporal dependencies
   - Learns nucleation dynamics end-to-end, enabling long-range forecasting
   - Requires no compressed time representations or injected future bubble positions

2. **Generalizing across fluids and flow regimes**
   - Conditions on thermophysical parameters for cross-liquid generalization (cryogenics, refrigerants, dielectrics)
   - Supports multiple boiling configurations (pool/flow boiling) and geometries (single/double-sided heaters)

3. **Physics-based evaluation**
   - Introduces interpretable metrics beyond pixel-wise error:
     - Heat flux divergence
     - Eikonal PDE for signed distance functions
     - Mass conservation
   - Evaluates physical correctness in chaotic systems

### Model Architecture

The primary model in Bubbleformer is:

- **FiLMAViT** : An Axial Vision Transformer with factored spacetime blocks, FiLM Layer for fluid parameter conditioning and techniques to preserve high frequency components.

## BubbleML 2.0 Dataset

BubbleML 2.0 is the most comprehensive boiling dataset to date, significantly expanding the original BubbleML with new fluids, boiling configurations, and flow regimes.

### Key Features

- **160+ high-resolution 2D simulations** spanning:
  - Pool boiling and flow boiling configurations
  - Diverse physics (saturated, subcooled, and single-bubble nucleation)
  - Three fluid types:
    - FC-72 (dielectric)
    - R-515B (refrigerant)
    - LN$_2$ (cryogen)

- **Experimental conditions**:
  - Constant heat flux boundary conditions
  - Double-sided heater configurations
  - Full range of flow regimes (bubbly, slug, annular until dryout)

### Technical Specifications

- **Simulation framework**: All simulations performed using Flash-X
- **Data format**: HDF5 files
- **Resolution**:
  - Spatial and temporal resolution varies by fluid based on characteristic scales
  - Adaptive Mesh Refinement (AMR) used where needed
  - AMR grids interpolated to regular grids with:
    1. Linear interpolation
    2. Nearest-neighbor interpolation for boundary NaN values

- **Contents**: Each simulation includes:
  - Temperature fields
  - Velocity components (x/y)
  - Signed distance functions (bubble interfaces)
  - Thermophysical parameters

For additional details on boundary conditions, numerical methods, and experimental validation, please refer to the bubbleformer paper Appendix B.

## Installation

You can use [uv](https://github.com/astral-sh/uv) to setup the python environment and install dependencies.

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

## Repository Structure

```
.
├── bubbleformer/              # Main package directory
│   ├── config/                # Configuration files
│   │   ├── data_cfg/          # Dataset configurations
│   │   ├── model_cfg/         # Model configurations
│   │   ├── optim_cfg/         # Optimizer configurations
│   │   └── scheduler_cfg/     # Learning rate scheduler configurations
│   ├── data/                  # Data loading and processing modules
│   ├── layers/                # Model layer implementations
│   ├── models/                # Model architecture implementations
│   └── utils/                 # Utility functions (losses, plotting, etc.)
├── env/                       # Environment configuration files
├── examples/                  # Example notebooks
├── samples/                   # Sample data files
└── scripts/                   # Training and inference scripts
```

## Usage

### Training

To train a model using the default configuration:

```bash
python scripts/train.py
```

To train with a specific configuration:

```bash
python scripts/train.py nodes=1 devices=1 max_epochs=400 batch_size=8
```

### Inference

The repository provides two ways to run inference:

1. Using the Python script:
```bash
python scripts/inference.py --model_path /path/to/model --data_path /path/to/data
```

2. Using the Jupyter notebook:
```bash
scripts/inference_autoregressive.ipynb
```