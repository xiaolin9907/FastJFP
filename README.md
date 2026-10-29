# FastJFP: Core Algorithms and Single-Case Examples

This repository contains the fast Jacobi fractional polynomial (JFP) algorithms and one fixed configuration for each example problem. It is self-contained and can be used independently of the manuscript workspace. Plotting, timing and storage benchmarks, parameter sweeps, raw experimental data, and batch runners are not included.

## Repository Layout

```text
Project.toml / Manifest.toml  Direct dependencies and locked environment
src/FastJFP.jl                Main loading entry point
src/                         Numerical algorithm modules
examples/common.jl           Loading, spectral inputs, and coefficient evaluation
examples/integral_action.jl   A single fractional integral action
examples/abel.jl              Abel integral equation
examples/multi_order.jl       Multi-order integral equation
examples/bbo.jl               BBO equation and physical solution reconstruction
examples/sdc.jl               Deferred correction in integral form
examples/eigenvalue.jl        Eigenvalues of the integral inverse operator
examples/airy.jl              Fractional Airy boundary-value problem
```

The `src/` directory retains the 11 numerical modules from the original project and adds the common entry point `FastJFP.jl`. Banded rational preconditioning is provided through `BandRational.jl`, which depends on both `AbelRational.jl` and `AiryRational.jl`. Relative loading paths between the other modules are unchanged.

`MathCompExamples.jl` retains the contour computation paths, with the complete-matrix comparisons and high-precision diagnostics removed. It therefore no longer requires `reference_mathcomp/Frac.jl` from the original workspace. The other 10 modules are copied unchanged. Integral actions, preconditioner construction, iterations, and reconstruction formulas in the retained paths are unchanged. `ReferenceJFP.jl` remains in `src/` as a loading dependency of the core code; complete-matrix comparison examples are not included.

## Environment and Usage

Use Julia **1.11.5**. Direct dependencies are BandedMatrices, FFTW, Krylov, KrylovKit, SpecialFunctions, and the required standard libraries. BenchmarkTools, plotting packages, and CSV packages are not required. `Manifest.toml` contains the necessary dependencies extracted from the original locked environment, with package versions preserved.

Clone the repository and install its dependencies:

```powershell
git clone https://github.com/xiaolin9907/FastJFP.git
cd FastJFP
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

Choose one example to run at a time. No command-line parameters or data files are required:

```powershell
julia --threads=1 --project=. --startup-file=no examples/integral_action.jl
julia --threads=1 --project=. --startup-file=no examples/abel.jl
julia --threads=1 --project=. --startup-file=no examples/multi_order.jl
julia --threads=1 --project=. --startup-file=no examples/bbo.jl
julia --threads=1 --project=. --startup-file=no examples/sdc.jl
julia --threads=1 --project=. --startup-file=no examples/eigenvalue.jl
julia --threads=1 --project=. --startup-file=no examples/airy.jl
```

These are seven independent commands, not a sequential test suite. Each script uses one fixed configuration, runs only the fast method, and prints the corresponding residuals, errors, or eigenvalues. It does not generate plots or files. Alternatively, use `include("examples/abel.jl")` in a Julia REPL and inspect `result` for the coefficients, right-hand side, and solution. Use a separate process for each example rather than repeatedly loading different entry points in the same REPL session.

## Example Equations and Parameters

The fractional integral operator has lower endpoint $-1$:

\[
 (\mathcal I_x^\mu f)(x)
 =\frac1{\Gamma(\mu)}\int_{-1}^{x}(x-s)^{\mu-1}f(s)\,ds.
\]

The basis is $t^\alpha T_j(2t^\beta-1)$, where $t=(1+x)/2$. All contour actions use $M=160$, with Float64/ComplexF64 arithmetic. Banded preconditioners for the linear equations use upper bandwidth 32. BLAS runs with one thread.

| Script | Fixed configuration | Computation |
|---|---|---|
| `integral_action.jl` | $N=128$, $\mu=\beta=1/2$, $\alpha=0$ | Evaluate $\mathcal I_x^{1/2}(1+x)$ and compare with $4(1+x)^{3/2}/(3\sqrt\pi)$ |
| `abel.jl` | $N=256$, $\mu=1/2$, $\alpha=0$, $\lambda=4$ | Solve the Abel equation with a spectral right-hand side; tolerance $10^{-14}$ |
| `multi_order.jl` | $N=30$, $(\alpha,\beta)=(0,1/6)$ | Solve the multi-order equation; tolerance $2\times10^{-14}$ |
| `bbo.jl` | $N=50$, $(\alpha,\beta)=(0,1/2)$ | Solve the auxiliary integral equation and reconstruct $v$; tolerance $2\times10^{-14}$ |
| `sdc.jl` | 10 mapped Chebyshev nodes | Apply deferred correction using contour integral actions; discrete residual tolerance $10^{-15}$ |
| `eigenvalue.jl` | $N=500$, $(\alpha,\beta)=(1/2,3/2)$ | Compute the 7 inverse eigenvalues of largest modulus; Arnoldi subspace dimension 40, tolerance $10^{-13}$ |
| `airy.jl` | $\epsilon=10^{-4}$, $N=680$, $(\alpha,\beta)=(1/2,1/2)$ | Use an auxiliary unknown; tolerance $10^{-13}$, at most 3 residual corrections |

### Abel

\[
 u(x)+2^{-\mu}\lambda(\mathcal I_x^\mu u)(x)=f_N(x).
\]

The `spectral_input` function in `common.jl` uses the original project's right-hand-side coefficient generation rule with seed 20260918. The first approximately $4N/5$ coefficients follow an exponentially decaying envelope, and the remaining coefficients retain a tail at the level of machine precision. The reported `solution.residual` is the recomputed relative coefficient residual.

### Multi-Order Equation

\[
 u+\sqrt{1+x}\,\mathcal I_x^{1/3}u
 +\mathcal I_x^{1/2}[(1+\cdot)^{1/3}u]=f,
\]

\[
 f(x)=(1+x)^{3/2}
 +\left[\frac{\Gamma(5/2)}{\Gamma(17/6)}
 +\frac{\Gamma(17/6)}{\Gamma(10/3)}\right](1+x)^{7/3}.
\]

The exact solution is $u=(1+x)^{3/2}$. The script reports the relative maximum error sampled at 2001 points in the polynomial coordinate.

### BBO

\[
 u+\mathcal I_x^{1/2}u+\mathcal I_x^1u=-1,
 \qquad v=1+\mathcal I_x^1u.
\]

This is the integral formulation of $v'+{}^{\mathrm C}\mathcal D_x^{1/2}v+v=0$ with $v(-1)=1$. The auxiliary unknown is $u=v'$. The script reconstructs $v$ and compares it with the analytic formula based on `erfcx` in the original code.

### SDC

\[
 u-\mathcal I_x^{1/2}u
 =\frac{\sqrt\pi}{2}(1+x)-\frac23(1+x)^{3/2}.
\]

The exact solution is $u=\sqrt\pi(1+x)/2$. The script retains the original piecewise-constant product-integration initialization and correction, and reports the discrete residual and the maximum absolute error at the nodes. The nodal values are not JFP coefficients.

### Eigenvalue Problem

\[
 u(x)+\lambda_{\rm eig}(\mathcal I_x^{3/2}u)(x)
 =\lambda_{\rm eig}\sqrt{\frac{1+x}{2}}(\mathcal I_x^{3/2}u)(1).
\]

This corresponds to $-\mathcal D_x^{3/2}u=\lambda_{\rm eig}u$ with $u(-1)=u(1)=0$. The eigenvalues of the integral inverse operator are $1/\lambda_{\rm eig}$. The script takes their reciprocals to return the eigenvalues of the original problem. High-precision eigenfunction diagnostics are not included.

### Airy

With $v=\mathcal D_x^{3/2}u$, the integral formulation is

\[
 \epsilon i^{3/2}v(x)-x(\mathcal I_x^{3/2}v)(x)
 -ax\sqrt{\frac{1+x}{2}}=0,
 \qquad a+(\mathcal I_x^{3/2}v)(1)=1.
\]

Reconstruct $u=\mathcal I_x^{3/2}v+a\sqrt{(1+x)/2}$, with boundary conditions $u(-1)=0$ and $u(1)=1$. Evaluation includes the endpoint weight with $\alpha=1/2$, so the script returns the physical solution rather than the regular factor with the weight removed. It reports coefficient residuals and boundary errors; these are not continuous solution errors.

## Packaging and Validation

This package was prepared by organizing the code and documentation, without installing dependencies, running examples, solving equations, or repeating existing experiments. Packaging checks covered Julia syntax, relative dependency paths, locked package versions, and consistency of the unchanged modules. These static checks do not constitute numerical revalidation. The original project's code, data, manuscript, and historical delivery packages were left unchanged.
