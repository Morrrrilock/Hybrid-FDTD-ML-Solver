# Hybrid-FDTD-ML-Solver

Hybrid FDTD–machine learning framework for autoregressive acoustic wavefield prediction with GPU acceleration.

---

# Overview

This repository presents a hybrid finite-difference time-domain (FDTD) and machine learning framework for acoustic wave propagation and autoregressive wavefield prediction.

The project combines:

* Physics-based FDTD simulation
* CNN-based autoregressive prediction
* GPU acceleration using CuPy
* Deep learning with PyTorch
* Temporal wavefield forecasting
* Scientific machine learning workflows

The framework first generates wavefield data using a standard 3D acoustic FDTD solver and then trains a convolutional neural network (CNN) to predict future wavefield evolution autoregressively.

---

# Governing Equation

The solver is based on the 3D linear acoustic wave equation:

<p align="center">
  <img src="./figures/wave_equation.png" width="420"/>
</p>

---

# Numerical Scheme

The solver employs a second-order explicit FDTD update scheme:

<p align="center">
  <img src="./figures/update_scheme.png" width="900"/>
</p>

The Courant number is defined as:

<p align="center">
  <img src="./figures/courant_number.png" width="250"/>
</p>

The CFL stability condition is:

<p align="center">
  <img src="./figures/courant_condition.png" width="220"/>
</p>

---

# Features

* 3D acoustic wave equation solver
* GPU acceleration with CuPy
* PyTorch-based CNN prediction model
* Autoregressive wavefield forecasting
* Temporal sequence learning
* Enhanced initial wave amplitudes
* Gaussian pressure bump initialization
* Ricker wavelet source
* Neumann boundary conditions
* Receiver signal analysis
* Prediction error evaluation
* Scientific visualization tools
* Hybrid physics–ML framework

---

# Hybrid FDTD–ML Workflow

The framework consists of four stages:

1. Generate wavefield data using FDTD simulation
2. Extract temporal wavefield slices
3. Train CNN on temporal sequences
4. Perform autoregressive future prediction

The CNN learns temporal wave propagation patterns directly from FDTD-generated wavefields.

---

# Machine Learning Architecture

The repository implements:

* Multi-layer 2D convolutional neural network
* Batch normalization
* ReLU activations
* Temporal autoregressive prediction
* Sequence-to-frame forecasting

Input:

* Previous wavefield snapshots

Output:

* Predicted future wavefield

---

# GPU Acceleration

GPU acceleration is implemented using:

* CuPy for FDTD simulation
* CUDA-enabled PyTorch for CNN training

If GPU acceleration is unavailable, the framework automatically falls back to CPU execution.

Supported platforms:

* NVIDIA CUDA GPUs
* Windows
* Linux
* Jupyter Notebook
* Google Colab

---

# Repository Structure

```text
Hybrid-FDTD-ML-Solver/
│
├── hybrid_fdtd_ml_solver.py
│   Main implementation of the hybrid FDTD–ML framework
│
├── README.md
│   Project documentation and usage instructions
│
├── requirements.txt
│   Python dependencies required for the project
│
├── figures/
│   Equation images and visualization figures
│   ├── wave_equation.png
│   ├── update_scheme.png
│   ├── courant_number.png
│   ├── courant_condition.png
│   └── autoregressive_results_enhanced.png
│
├── results/
│   Generated prediction and benchmarking results
│
└── .gitignore
    Git ignore rules for environments and cache files
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Hybrid-FDTD-ML-Solver.git
cd Hybrid-FDTD-ML-Solver
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main dependencies:

```text
numpy
matplotlib
cupy-cuda12x
torch
scipy
tqdm
```

---

# Usage

Run the complete hybrid FDTD–ML demo:

```bash
python hybrid_fdtd_ml_solver.py
```

The script will:

1. Generate FDTD wavefield data
2. Train CNN prediction model
3. Run autoregressive forecasting
4. Evaluate prediction accuracy
5. Generate visualization figures

---

# Example Output

The framework produces:

* Wavefield prediction results
* Receiver signal comparisons
* Prediction error curves
* Signal envelope analysis
* CNN autoregressive forecasts

Generated figures include:

* FDTD vs CNN prediction comparison
* Error evolution
* Wave amplitude analysis
* Temporal forecasting performance

---

# Applications

Potential applications include:

* Computational acoustics
* Scientific machine learning
* Neural PDE surrogate modeling
* Wave propagation forecasting
* Hybrid physics–AI simulation
* Acoustic modeling
* Reduced-order modeling
* Neural simulation acceleration

---

# Future Work

Planned extensions:

* 3D CNN architectures
* Transformer-based wave prediction
* Multi-step forecasting
* Physics-informed neural networks (PINNs)
* Differentiable FDTD
* Absorbing boundary conditions (PML)
* Binaural acoustic simulation
* Hybrid FDTD–transformer frameworks

---

# License

This project is released under the MIT License.

---

# Author

Developed by Zongwen Hu for research in:

* Computational acoustics
* Scientific computing
* GPU acceleration
* Scientific machine learning
* Wave propagation simulation

---
