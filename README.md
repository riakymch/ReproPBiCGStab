# ReproPBiCGStab

## Introduction

ReproPBiCGStab aims to ensure reproducibility and accuracy of the pure MPI implementation of the preconditioned Bi-conjugate Gradient stabilized method. The project is composed of the following branches:
- orig containing the original version of the code

- master with the reproducible and accurate implementation using the ExBLAS approach

- opt_exblas optimizes the previous version by relying only on floating-point expansions (FPE, short arrays of doubles) with error-free transformations (`twosum` and `twoprod`). This version employs FPEs of size 8 with the early-exit technique

- mpfr provides highly accurate sequential implementation using the MPFR library. It serves as a reference

Currently, we also consider to apply vectorization to the opt_exblas using the VCL library

## Installation

#### Requirements:
- `c++ 11` (presently with `gcc-8.2.0` or higher)

- support of fma, especially for the opt_exblas branch

- [optional] separate installation of the MPFR library

#### Building ReproPBiCGStab

1. clone the git-repository into `<ReproPBiCGStab_root>`

2. inside the src directory, invoke `make` to create BiCGStab executable

## Example
The code can be run using two modes
- matrix from the Suite Sparse Matrix Collection

`mpirun -np P --bind-to core ./ReproPBiCGStab/src/BiCGStab MAT.rb 1`
 
