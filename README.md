# Learning freely-described convex sets from data
This is the MATLAB implementation of the algorithm from [paper] for learning convex sets instantiable in any dimension from data.

## Prerequisites
Please install these packages and add them to MATLAB's path.
1. [YALMIP](https://yalmip.github.io/download/) with an SDP solver like [MOSEK](https://www.mosek.com/downloads/)
2. [Package to find null space of a sparse matrix](https://www.mathworks.com/matlabcentral/fileexchange/11120-null-space-of-a-sparse-matrix)

## Main functions
1. Script to learn an SDP approximation of $\ell_p$ norms: [LpNorm_learn](/LpNorm_learn.m)
2. Script to learn an SDP approximation of (variant of) quantum entropy: [quantEntropy_learn](/quantEntropy_learn.m)
