# ckt_PSE  
**Circuit-Theoretic Parameter–State Estimation**  
_Last updated: 31 Oct 2023_

This repository contains the final implementation of a **circuit-theoretic state estimation (SE) and joint parameter–state estimation (PSE)** framework for power systems. The method formulates SE/PSE as a nonlinear optimization problem and solves it using **Pyomo + IPOPT**.

This branch represents the **final research version** of the project.

---

## Overview

### Main Features
- Circuit-theoretic formulation of power system state estimation  
- Joint estimation of:
  - Bus voltage states  
  - Selected unknown branch or shunt parameters  
- Support for:
  - Synthetic measurement generation  
  - Additive Gaussian noise  
  - Convex relaxations (McCormick, QC, cone)  
  - Sequential bound tightening (SBT)  
- Nonlinear optimization solved using **IPOPT**

---

## Dependencies

### Operating System
- **Linux / WSL (Ubuntu recommended)**  
  The code is developed and tested under WSL.

### Required Software
- **Python 3.8 or newer**
- **IPOPT** (nonlinear optimization solver)

### Required Python Packages
- `pyomo`
- `numpy`

Additional scientific packages such as `scipy`, `pandas`, and `matplotlib` may be used depending on configuration.

### Repo Structure
```bash
ckt_PSE/
├── code/
│   ├── run_solver.py          # Main entry point
│   ├── testcases/             # RAW network files
│   └── scripts/
│       ├── Solve.py           # Core solver logic
│       ├── options.py         # Configuration file
│       └── IPOPT_workfile.py  # Additional IPOPT settings
├── requirements.txt
└── README.md
```bash

### Recommended Installation (Conda)
```bash
conda create -n cktpse python=3.10 -y
conda activate cktpse
conda install -c conda-forge pyomo ipopt numpy -y
```bash
  ## Configuration (`scripts/options.py`)

All major configuration parameters are defined in `scripts/options.py` under a single `OPTIONS` dictionary with the following blocks:

### Power flow / simulation
`OPTIONS["powerflow"]`
- `load_factor`: scales system loading used in simulation (can affect current-related terms and sensitivity in parameter estimation)
- `settings`: power flow solver settings (`Tolerance`, `Max Iters`, `Limiting`, `Sparse`)

### Unknown parameter estimation
`OPTIONS["unknown_params"]`
- `unknown_branch_B`, `unknown_B_branch_id`: enable/define unknown branch susceptance(s)
- `unknown_branch_G`, `unknown_G_branch_id`: enable/define unknown branch conductance(s)
- `unknown_branch_sh`, `unknown_sh_branch_id`: enable/define unknown shunt parameters
- `unknown_transformer_tr`, `unknown_tr_transformer_id`: enable/define unknown transformer parameter(s)

> If you are mainly working on standard state estimation (SE), keep all unknown-parameter flags set to `False`.

### Objective weights
`OPTIONS["weights"]`
Defines relative weights for different residual components (e.g., slack, nv, nr, ni). These weights can significantly affect parameter estimation sensitivity.

### Solver (IPOPT)
`OPTIONS["solver"]`
- `tolerance`, `max_iter`: IPOPT convergence and iteration limits
- `tee`: print solver iteration logs
- `Obj_scal_factor`: objective scaling factor (used to improve numerical conditioning)
- `model_type`: selects between internal model variants (1–3 main, 4 test-only)

### Relaxation / tightening
`OPTIONS["relaxation"]`
- `flag_McCormick`: enable McCormick relaxation for bilinear terms (typically used when unknown parameters exist)
- `non_convex_equality`: keep quadratic voltage magnitude equality constraints
- `qc_relaxation`, `cone_relaxation`: optional convex relaxations
- `tighten_B`: sequential bound tightening (SBT) for unknown B
- `B_ineq_constraint`: enforce inequality constraints on B if enabled

### Measurement
`OPTIONS["measurement"]`
- `flag_noise`: enable additive Gaussian noise
- `same_noise`, `random_seed`: reproducibility controls
- `distribution`: noise parameter (see code for whether this is interpreted as σ or variance)
- `RTU`, `PMU`: measurement type toggles

