# ckt_PSE (Last updated at 31/10/2023)
This branch is the final branch of the state estimation project.
# How to run this code
Run the run_solver.py file:
* Note the code is tested and run on WSL 
* Before running the code, please make sure the following packages are correctly installed: Anaconda, Pyomo, and ipopt.
* Casename: fits the testcase.RAW file
* Settings: set the powerflow simulation options
* When SBT is activated, please make sure that the best-known objective is checked to be feasible (you can put the objective as a constraint into NLP solver and run it to see if it is a feasible one) or SBT will give all infeasible solutions and solve as an NLP. 
# Options
In scripts you will find options.py, you can modify the following settings of the problem:
1. In the measurement function where the synthetic measurement is generated:
* flag_noise (bool) True to produce noisy measurement
* same_noise (bool) True if the same random seed is required to generate the noise term
* get_new-seed (bool) Generate a new random seed (Not in use in this version)
* random_seed (int) Chose random seed for same_noise
* distribution (flot) Set the sigma of the normal distribution where the noise term is generated
2. The optimization problem is solved with ipopt solver {citation}. The following options for ipopt and Pyomo are available in options.py:
* max_iter (int) Maximum number of ipopt iterations
* tolerance (float) Desired ipopt convergence tolerance
* flag_tee (bool) Enables the solver logs when True
* G_noise (bool) Enables a different V_r*n_r noise.
* Also, another setting that is not in options.py can be found in script/IPOPT_workfile.py
3. The unknown parameter setting can replace the chosen known parameter with a variable:
* unknown_branch_B (bool) True if an unknown branch parameter exists
* unknown_B_id (list) The list of unknown branches, corresponding to the branch id provided in .RAW file.
* unknown_shunt_id (list) The list of unknown shunts, corresponding to the shunt id provided in .RAW file.
4. Validation methods can be chosen when unknown_branch_B is True:
* flag_McCormick (bool) True if MC-convex relaxation of the bilinear term (state_variable* unknown_parameter)
* tighten_B (bool) True if apply SBT to unknown branch susceptance B
* tighten_shunt_B (bool) True if apply SBT to unknown shunt susceptance B
* non_convex_equality (bool, when flag_McCormick is True) True if quadratic voltage measurement constraint is kept
* qc_relaxation (bool, when flag_McCormick is True) True if QC-relaxation is applied to quadratic voltage measurement constraint

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

