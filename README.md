# Learning freely-described convex sets from data
This is the MATLAB implementation of the algorithm from [paper] for learning convex sets instantiable in any dimension from data.

## Prerequisites
Please install these packages and add them to MATLAB's path.
1. [YALMIP](https://yalmip.github.io/download/) with an SDP solver like [MOSEK](https://www.mosek.com/downloads/)
2. [Package to find null space of a sparse matrix](https://www.mathworks.com/matlabcentral/fileexchange/11120-null-space-of-a-sparse-matrix)

## Main functions
1. Script to learn an SDP approximation of $\ell_p$ norms: [LpNorm_learn](https://github.com/eitangl/anyDimCvxSets/blob/main/LpNorm_learn.m).
2. Script to learn an SDP approximation of (variant of) quantum entropy: [quantEntropy_learn](https://github.com/eitangl/anyDimCvxSets/blob/main/quantEntropy_learn.m).
3. Script to verify the graphon generation degree in Prop. 3.1: [check_graphon_gen_deg](https://github.com/eitangl/anyDimCvxSets/blob/main/check_graphon_gen_deg.m)
4. Scripts to compute dimensions for spaces of invariants / morphisms for the examples in Sec. 4.1: [compute_dims_a](https://github.com/eitangl/anyDimCvxSets/blob/main/compute_dims_a.m), [compute_dims_b](https://github.com/eitangl/anyDimCvxSets/blob/main/compute_dims_b.m), and [compute_dims_c](https://github.com/eitangl/anyDimCvxSets/blob/main/compute_dims_c.m).

In case of issues or questions, please email Eitan (eitanl@caltech.edu)
