# NPEFF Reference Implementations
<!-- Reference implementation of NPEFF stuff. -->

<!-- TODO: Link paper once on arxiv. -->
This repo contains implementations of some of the methods introduced in *NPEFF: Non-Negative Per-Example Fisher Factorization*.

## Overview

### Cuda/C++
Cuda/C++ code is contained within the `cc` directory.
The dictionary learning algorithms are entirely implemented in cuda/C++.
Besides that, everything else is implemented in Python.

The `cc` directory has two sub-directories: `cuda_d_npeff` and `cuda_lrm_npeff`.
Each of these have their own `src` and `mains` sub-directories that contain library code
and files that compile into executables, respectively.
The `cuda_d_npeff` directory implements D-NPEFF algorithms, and the `cuda_lrm_npeff`
directory implements LRM-NPEFF algorithms.
This separation is an artifact of how the research project progressed.
There's a decent amount of code duplication as well, so these libaries could be unified
and cleaned up at some point in the future.

We use CMake as the build system.
Most of the dependencies should be included in the standard Cuda installation.
However, you will need to [install NCCL](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down).
We use NCCL to handle multiple GPUs, but it will need to be installed even if you are running
on a single GPU.

You will also need to install some hdf5 libraries via something like:
```bash
sudo apt install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
```

Once these steps have been completed, the C++ can be compiled via:
```bash
# Enter the directory. Do the same for cuda_lrm_npeff.
cd ./cc/cuda_d_npeff

# Configure. This only needs to be done once.
cmake -S . -B build

# Build the code. Needs to be done whenever changes are made to the code.
cmake --build build
```

The compiled executables will be contained in the `./cc/cuda_d_npeff/build/mains` and `./cc/cuda_lrm_npeff/build/mains`
directories.
Here is an overview of the available executables.
For information on their flags, please see their corresponding source files.

#### D-NPEFF
- `run_nmf_on_pefs`: Runs NMF on a matrix composed of D-PEFs. 
- `fit_coeffs_to_sparse_H`: Fits NPEFF coefficients to a set of D-PEFs given a D-NPEFF decomposition. The decomposition must have been sparsified first via the `scripts/sparsify_d_npeff.py` script.


#### LRM-NPEFF
- `run_m_npeff`: Performs the dictionary learning procedure on a set of LRM-PEFs. Each LRM-PEF must have the same rank.
- `fit_m_npeff_coeffs`: Fits NPEFF coefficients to a set of LRM-PEFs given an LRM-NPEFF decomposition. Each LRM-PEF must have the same rank.
- `run_m_npeff_expansion`: Performs a dictionary expansion procedure on a set of LRM-PEFs. Each LRM-PEF must have the same rank.
- `run_lvrm_npeff`: Performs the dictionary learning procedure on a set of LRM-PEFs. The LRM-PEFs can have different ranks for different examples.
- `fit_lvrm_coeffs`: Fits NPEFF coefficients to a set of LRM-PEFs given an LRM-NPEFF decomposition. The LRM-PEFs can have different ranks for different examples.


### Python
Python code is contained in the `npeff` and `scripts` directories.
They contain library code and scripts, respectively.

Here is an overview of the available scripts within the `scripts` directory.
For information on their flags, please see their corresponding Pythong files.

- `compute_d_dsf.py`: Computes the diagonal approximation of a dataset-level Fisher matrix.
- `compute_pefs.py`: Computes and save PEFs to an hdf5 file. Flags allow for computation of either D-PEFs or LRM-PEFs.
- `filter_pefs.py`: Filters a saved set of PEFs to create a new set of PEFs satisfying some critera. Currently, only filtering to create a set of PEFs corresponding to examples the model made incorrect predictions for is supported.
- `make_top_examples_collages.py`: Makes collages of top component examples for NPEFFs of image models.
- `make_top_examples_latex.py`: Makes a LaTeX document containing top component examples for NPEFFs of text models.
<!-- - `perturb_d_npeff_components.py`:  -->
- `perturb_lrm_npeff_components.py`: Runs perturbation experiments for LRM-NPEFF decompositions.
- `sparsify_d_npeff.py`: Converts the H matrix of a D-NPEFF decomposition into a sparse representation.


<!-- ## Example Workflows -->

