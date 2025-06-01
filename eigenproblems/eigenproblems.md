# Problem setup

Consider linear parametric eigenproblem
```math
\mathcal{A}(x; \omega)u_{i}(x; \omega) = \lambda_i(\omega) u_i(x; \omega),\,\left\|u_i\right\| = 1,
```
where $\omega$ is a set of parameters and $\mathcal{A}$ is a linear positive semi-definite operator. Our usual example is
```math
\text{div }k_{\omega}(x)\text{ grad }u_i(x; \omega) = \lambda_i(\omega)u_i(x; \omega)\in\Gamma=[0, 1]^2,\,\left\|u_i\right\| = 1,\,\left.u_{i}(x; \omega)\right|_{x\in\partial\Gamma} = 0.
```
We sample a set of parameters $\omega_1, \dots, \omega_N$ and find $K$ (typically $K\leq 100$) low-lying  eigenmodes $V_{1}, \dots, V_{N}$. Our goal is to produce computationally cheap surrogate that can predict eigenvectors for new parameters $\omega$ not present in the dataset.

We consider two ways to solve this problem:
1. Direct regression: Train parametric model to predict $V$ given $\omega$
2. Dimension reduction: Train parametric model to predict subspace $W$ such that $\text{range }W = \text{range }V$ and use this subspace in reduced $K\times K$ eigenproblem (Petrov-Galerkin).

# Loss functions