# Danaher, Wang and Witten (2014): three-group FGL simulation

This example recreates the simulation **design** behind Figure 2, Section 7.1
of [The joint graphical lasso for inverse covariance estimation across multiple
classes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4012833/), using
`CovarianceSelection.fusedGraphicalLasso`. It compares FGL at four levels of
fusion with separate graphical lassos, the same solver with zero fusion.
It writes every replicate, Monte Carlo means and standard errors, and the
five types of comparison plotted in the paper. The original graph, random
seeds and complete sparsity grid were not available, so this is not an exact
reproduction of the published numerical values. The GGL and Guo et al.
curves, the two-group experiments, and the lung-cancer analysis are outside
this example's scope.

[Saved results](results/README.md) include a completed five-replicate pilot
at the paper's dimensions, raw metrics, the plots and a six-fit R/JGL check.

## Run

Use the example environment described in [../README.md](../README.md).
From the repository root:

```sh
# Small demonstration: p=100, n=150 per group, three replicates.
julia --project=examples examples/danaher2014.jl

# Paper dimensions, initially one replicate at lambda1=0.175.
julia --project=examples examples/danaher2014.jl --mode=pilot

# Paper dimensions and 100 replicates, with the four Figure 2 fusion values.
# This runs 6,500 fits, including separate graphical lassos; it can take hours.
julia --project=examples examples/danaher2014.jl --mode=paper

# Recreate the saved five-replicate pilot (100 fits).
julia --project=examples examples/danaher2014.jl --mode=pilot --reps=5 \
  --lambda1=0.30,0.225,0.175,0.1 --lambda2=0.025,0.05,0.075,0.1 \
  --out=examples/danaher2014/output/figure2-pilot
```

`--mode=quick` uses p=50, n=40 and six fits, as does
`COVSEL_EXAMPLES_QUICK=1`. Every option uses `--name=value`:
`mode`, `p`, `n`, `reps`, `seed`, `lambda1`, `lambda2`, `out`, and
`export-data`. Comma-separated penalty lists override the preset grid;
zero fusion is always included. `p` must be a multiple of ten and at least
20. The default seed is 2014. A run refuses to overwrite existing replicate
results; give each run its own `--out` directory.

## Simulation and penalty conventions

- K=3, with ten equal-sized disconnected generating subnetworks. A linear
  preferential-attachment tree with one edge added per new vertex implements
  the paper's unspecified power-law generator. At p=500 the generating
  graphs have 490, 441 and 392 edges: eight blocks are shared by all groups,
  one by groups 1 and 2, and one belongs to group 1 alone.
- Generate symmetric edge weights uniformly on `[-0.4,-0.1] ∪ [0.1,0.4]`.
  Divide each off-diagonal row by 1.5 times its absolute row sum and
  symmetrize. Invert this matrix, normalize it to a correlation matrix and
  multiply off-diagonal covariances by 0.6, following Section 7.1.1. The
  symmetrized matrix is checked for positive definiteness: failing block
  draws are resampled, without a ridge correction. The number rejected is
  recorded. The printed row-normalization recipe alone does not guarantee
  positive definiteness for every preferential-attachment tree.
- Hold the population fixed across Monte Carlo samples. Each replicate uses
  `Xoshiro(seed + replicate)`; the population uses `Xoshiro(seed)`. Draw
  independent Gaussian observations, center each group and use the sample
  covariance with divisor **n**, matching the authors' R implementation.
  Data generation occurs once per replicate and is shared by every fit.
- Section 7 multiplies the **displayed** penalties by n in the n-weighted
  objective. Consequently this example uses `weights=ones(3)` and the
  displayed penalties unchanged. With the solver's default `weights=1/3`,
  divide both displayed penalties by **3**, not by `3n`. The diagonal is
  excluded from the lasso penalty and retained in the fusion penalty.
- The `paper` profile uses the Figure 2 legend's fusion values
  `0.025, 0.05, 0.075, 0.1`. Its sparsity grid is explicitly chosen here:
  `0.35, 0.325, ..., 0.05`; the original complete sparsity grid is not
  specified in the paper. The profile uses p=500, n=150 and 100 replicates.

There is a material distinction between the sparse **generating graph** and
the actual precision matrix from the printed construction. In general,
`Σ = 0.6 cor(A⁻¹) + 0.4I` has an inverse with extra nonzero entries within
each block. Thus its actual precision support need not have the stated 490
edges. The primary `tp`, `fp`, `tp_diff`, `fp_diff` metrics use the actual
`Θ=Σ⁻¹`, as the paper defines its recovery criteria; `graph_*` columns
separately evaluate the original sparse skeleton. `metadata.txt` records
both edge counts. Neither population precision nor generating block labels
are passed to the solver; it fits the full empirical covariance matrices.

## Outputs and plots

The default destination is `examples/danaher2014/output/MODE/`, ignored by Git.

- `replicates.tsv`: one row per fit, including convergence iteration count,
  fit time, and relative proximal fixed-point residual.
- `summary.tsv`: means and Monte Carlo standard errors over replicates.
  One-replicate standard errors are `NaN`.
- `metadata.txt`: dimensions, grid, seed, covariance and threshold
  conventions, solver tolerances, Julia/BLAS information and scope.
- With `--export-data=true`, first-replicate data and fitted precision
  matrices for the independent R comparison.

The five panels are true versus false selected edges, off-diagonal squared
error versus selected edges, true versus false differential edges, dKL
versus off-diagonal L1 norm, and elapsed fit time versus selected edges.
Undirected edges and group-pair differences count each `i<j` once; SSE and
L1 count both off-diagonal orientations. Estimated edges use `1e-5`, actual
precision truth uses `1e-8`, FGL differences use `1e-6` to allow numerical
error, and the separate graphical lasso uses the paper's `1e-2` difference
threshold. The paper's displayed dKL omits the usual constant `Kp/2`; both
`dkl_paper` and ordinary zero-at-truth `kl` are saved. The latter is
`(tr(Θhat Σ) - logdet(Θhat Σ) - p)/2` summed across groups.

All fits must converge and have relative proximal fixed-point residual
below `1e-5`; a failure stops the run. Raw results are saved after every
completed fit. Fits start from zero, after a small compilation warm-up, and
run with one BLAS thread. Times describe this implementation and hardware;
they do not reproduce the paper's historical timing experiment or its
block-screened R implementation.

Generate standalone PNG and SVG figures with Python and matplotlib:

```sh
python3 -m pip install matplotlib  # once, preferably in a virtual environment
python3 examples/danaher2014/plot.py examples/danaher2014/output/figure2-pilot
python3 examples/danaher2014/plot.py examples/danaher2014/output/figure2-pilot --generating-graph
```

The error bars are one Monte Carlo standard error. A small pilot need not
recover the paper's method rankings; the full grid is intended for
investigating them, with all thresholds and regeneration choices explicit.

## Independent R/JGL check

```sh
julia --project=examples examples/danaher2014.jl --mode=quick \
  --export-data=true --out=examples/danaher2014/output/reference
Rscript examples/danaher2014/compare_jgl.R examples/danaher2014/output/reference
```

The R script downloads five MIT-licensed source files from the authors'
[JGL 2.3.2 CRAN release](https://github.com/cran/JGL/tree/52fc1717ad075c0478330b029d005eaaf1f52690),
pinned to that commit, and calls its ADMM directly on the same exported
observations. No R packages are needed. This avoids differences from
screening, truncation, centering and penalty scaling in high-level wrappers.
It checks maximum entry differences below `1e-3` and relative objective
differences below `1e-6`, and writes `jgl_reference.tsv`. The R all-pairs
proximal implementation pools near ties at `1e-4`, so it cannot serve as an
exact reference at finer tolerances. This check validates solver agreement,
not equivalence of the regenerated population to the authors' original data.
