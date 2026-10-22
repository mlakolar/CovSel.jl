#!/usr/bin/env Rscript
# bench_r.R -- run an R implementation of CLIME on a given covariance matrix, for the
# comparison in README.md (packages installed by install_r.R).
#
# Usage:
#   Rscript bench_r.R <cov.bin> <p> <lambda_grid.txt> <method> <out_prefix> [key=value ...]
#
#   cov.bin          raw little-endian Float64, p*p values, column-major (symmetric covariance)
#   p                dimension
#   lambda_grid.txt  text file, one lambda per line, decreasing
#   method           fastclime      fastclime::fastclime (parametric simplex path), exact LP solution
#                                   at each grid lambda obtained by linear interpolation between the
#                                   two path breakpoints that bracket lambda
#                    flare          flare::sugm(method = "clime") (linearized ADMM, multistage screening)
#                    clime          clime::clime(linsolver = "primaldual") (package default; l1-magic
#                                   primal-dual interior point)
#                    clime_simplex  clime::clime(linsolver = "simplex") (lpSolve, one LP per column/lambda)
#   out_prefix       prefix of all output files (directory must exist)
#
# Options (key=value):
#   lib=<dir>            library holding the packages (default: <dir of this script>/Rlib)
#   cols=TRUE|FALSE      write unsymmetrized column solutions (default TRUE)
#   diag=TRUE|FALSE      per-column constraint/l1 diagnostics (default TRUE; needs column solutions)
#   reps=<int>           repeat the timed solve (default 1); outputs come from the last repetition
#   fc_nlambda=<int>     fastclime path-length cap (pivots per column) for the first try (default max(100, 2p));
#                        multiplied by fc_grow and re-run (at most fc_maxtries times) while some column is
#                        truncated. Memory: fastclime() holds ~4-5 copies of an nlambda x p^2 double array
#                        (4.8 GB peak at p = 400, nlambda = 800); fc_direct=TRUE needs ~2 copies.
#                        AR(1) n = 200, lambda.min = 0.1 needed max 74 / 333 / 1127 pivots for p = 100/200/400.
#   fc_maxtries=<int>    (default 4)
#   fc_conly=TRUE|FALSE  also time the bare .C("parametric") call (default TRUE)
#   fc_grow=<num>        growth factor of nlambda between tries (default 2)
#   fc_direct=TRUE|FALSE call the C routine of fastclime() directly with the same arguments, skipping the
#                        wrapper's extra copies of the nlambda x p^2 path array (default FALSE; timing row
#                        "fastclime_direct"); use when the wrapper would need too much memory
#   flare_prec=<num>     sugm prec (default 1e-4 = package default)
#   flare_maxite=<int>   sugm max.ite (default 1e4 = package default)
#   flare_rho=<num>      sugm rho (default 1 = package default)
#   pdtol=<num>          clime primal-dual duality-gap tolerance (default 1e-3 = package default)
#   pdmaxiter=<int>      clime primal-dual max iterations (default 50 = package default)
#   clime_mode=api|internal
#                        api (default): time clime::clime(); then, if cols=TRUE, recompute the column
#                        solutions with the package's internal per-column solvers (clime:::linprogS /
#                        clime:::linprogPD, same calls as inside clime()), timed separately.
#                        internal: time a verbatim copy of clime()'s solve loop that keeps the columns
#                        (one solve instead of two; checked to reproduce clime() exactly).
#   timelimit=<sec>      elapsed-time limit for the R-level solve loops (default Inf). C code (fastclime,
#                        one flare column, one lpSolve call) is not interruptible; use an external alarm.
#   write=TRUE|FALSE     write the estimates and column solutions as .bin files (default TRUE); the
#                        timing and diagnostics files are always written
#
# All methods are run on the covariance matrix exactly as given: no standardization, no perturbation.
#   fastclime: has no such options; a square symmetric x is treated as the covariance (isSymmetric).
#   flare:     sugm(S, method = "clime", standardize = FALSE, perturb = FALSE)  [defaults: FALSE, TRUE;
#              perturb = TRUE adds 1/sqrt(n) I, and n = nrow(S) = p for covariance input]
#   clime:     clime(S, sigma = TRUE, perturb = FALSE)  [default perturb = TRUE adds
#              max(ev_max - p ev_min, 0)/(p-1) I; standardize only applies to data input]
#
# Outputs:
#   <prefix>_timing.csv           one row per timed run: method, p, K, rep, elapsed, user, system, status, details
#   <prefix>_lam<k>.bin           symmetrized estimate at lambda_k (k = 1..K, grid order), Float64 column-major
#   <prefix>_cols_lam<k>.bin      unsymmetrized column solutions at lambda_k (column j solves the LP for e_j)
#   <prefix>_selector_lam<k>.bin  fastclime only: fastclime.selector() output (package's own answer = the
#                                 path solution at the first breakpoint <= lambda_k, symmetrized)
#   <prefix>_diag.csv             per column: l1 norm, max|S b - e_j|, relative violation, nnz
# Symmetrization (all packages): keep the smaller-magnitude entry of each (i,j)/(j,i) pair.
# A column that could not be computed (e.g. infeasible LP) is written as NaN.

usage <- "Usage: Rscript bench_r.R <cov.bin> <p> <lambda_grid.txt> <fastclime|flare|clime|clime_simplex> <out_prefix> [key=value ...]\n"
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) { cat(usage); quit(status = 2) }
covf <- args[1]; p <- as.integer(args[2]); gridf <- args[3]; method <- args[4]; prefix <- args[5]
if (!method %in% c("fastclime", "flare", "clime", "clime_simplex")) { cat("unknown method\n", usage); quit(status = 2) }

opts <- list()
for (a in args[-(1:5)]) {
  kv <- regmatches(a, regexpr("=", a), invert = TRUE)[[1]]
  if (length(kv) != 2) stop("bad option: ", a)
  opts[[kv[1]]] <- kv[2]
}
opt <- function(key, default, conv = identity) if (is.null(opts[[key]])) default else conv(opts[[key]])
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(script_file)) dirname(normalizePath(script_file)) else getwd()
lib <- opt("lib", file.path(script_dir, "Rlib"))
.libPaths(c(lib, .libPaths()))
want_cols <- opt("cols", TRUE, as.logical)
want_write <- opt("write", TRUE, as.logical)
want_diag <- opt("diag", TRUE, as.logical)
reps <- opt("reps", 1L, as.integer)
timelimit <- opt("timelimit", Inf, as.numeric)

## ---- input ---------------------------------------------------------------------------------
sz <- file.info(covf)$size
if (is.na(sz) || sz != 8 * p * p) stop(sprintf("%s has %s bytes, expected 8*p*p = %d", covf, sz, 8 * p * p))
S <- matrix(readBin(covf, "double", n = p * p, size = 8, endian = "little"), p, p)
asym <- max(abs(S - t(S)))
if (asym > 1e-10 * max(abs(S))) warning(sprintf("input not symmetric (max |S - S'| = %.3g); symmetrizing", asym))
S <- (S + t(S)) / 2          # bitwise no-op for an exactly symmetric input
lambda <- as.numeric(trimws(readLines(gridf)))
lambda <- lambda[!is.na(lambda)]
K <- length(lambda)
if (K == 0) stop("empty lambda grid")
if (any(diff(lambda) > 0)) stop("lambda grid must be decreasing")
if (any(lambda <= 0)) stop("lambda must be positive")

symmetrize_min <- function(M) M * (abs(M) <= abs(t(M))) + t(M) * (abs(M) > abs(t(M)))
write_bin <- function(M, f) { con <- file(f, "wb"); writeBin(as.vector(M), con, size = 8, endian = "little"); close(con) }
timing_rows <- list()
add_timing <- function(meth, rep, tm, status = "ok", details = "") {
  timing_rows[[length(timing_rows) + 1]] <<- data.frame(
    method = meth, p = p, K = K, rep = rep,
    elapsed = if (is.null(tm)) NA else unname(tm[["elapsed"]]),
    user = if (is.null(tm)) NA else unname(tm[["user.self"]] + tm[["user.child"]]),
    system = if (is.null(tm)) NA else unname(tm[["sys.self"]] + tm[["sys.child"]]),
    status = status, details = details, stringsAsFactors = FALSE)
}
flush_timing <- function() write.csv(do.call(rbind, timing_rows), paste0(prefix, "_timing.csv"), row.names = FALSE)
quiet <- function(expr) invisible(capture.output(expr, file = nullfile()))
timed <- function(expr) {           # elapsed/user/sys of evaluating expr in the caller's frame
  e <- substitute(expr); pf <- parent.frame()
  gc(FALSE)
  setTimeLimit(elapsed = timelimit, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf))
  system.time(eval(e, pf), gcFirst = FALSE)
}

sym_list <- vector("list", K)   # symmetrized estimates
col_list <- NULL                # unsymmetrized column solutions (if available)
blas <- tryCatch(basename(extSoftVersion()[["BLAS"]]), error = function(e) "?")
base_details <- sprintf("R=%s;BLAS=%s", getRversion(), if (nzchar(blas)) blas else "Rblas")

## ---- fastclime -------------------------------------------------------------------------------
if (method == "fastclime") {
  suppressPackageStartupMessages(library(fastclime))
  lmin <- min(lambda)
  nl <- opt("fc_nlambda", max(100L, 2L * p), as.integer)
  maxtries <- opt("fc_maxtries", 4L, as.integer)
  grow <- opt("fc_grow", 2, as.numeric)
  direct <- opt("fc_direct", FALSE, as.logical)
  # direct=TRUE: call the same C routine with the same arguments as fastclime() does, but keep the raw
  # nlambda x p^2 path array instead of the wrapper's extra copies (unlist/matrix/icovlist): ~2 copies of
  # nlambda*p^2 doubles at peak instead of ~4-5.
  run_path <- function(nl) {
    if (!direct) {
      quiet(fit <- fastclime(S, lambda.min = lmin, nlambda = nl))
      LM <- fit$lambdamtx; if (is.null(dim(LM))) LM <- matrix(LM, ncol = p)
      list(LM = LM, maxnl = fit$maxnlambda, nl = nl, fit = fit,
           sol = function(k, j) fit$icovlist[[k]][, j])
    } else {
      str <- .C("parametric", as.double(S), as.integer(p), double(nl * p), as.double(lmin), as.integer(nl),
                as.integer(0), double(as.numeric(nl) * p * p), PACKAGE = "fastclime")
      maxnl <- str[[6]] + 1L
      LM <- matrix(str[[3]], nl, p)[seq_len(maxnl), , drop = FALSE]
      iic <- str[[7]]; rm(str)
      off <- nl * (0:(p - 1))
      list(LM = LM, maxnl = maxnl, nl = nl, fit = NULL,
           sol = function(k, j) iic[k + nl * (j - 1) * p + off])
    }
  }
  path_info <- function(LM, nl) {
    L <- apply(LM, 2, function(m) { i <- which(m != 0 & is.finite(m)); if (length(i)) max(i) else 0L })
    last <- LM[cbind(pmax(L, 1L), seq_len(p))]
    trunc <- (L >= nl) & (last > lmin)                 # ran out of pivots before reaching lmin
    infeas <- (L < nl) & (last > lmin) & (last > 1e-5) # ratio test failed: LP infeasible below 'last'
    list(L = L, last = last, trunc = trunc, infeas = infeas)
  }
  for (r in seq_len(reps)) {
    tries <- 0L; nl_r <- nl; total_el <- 0
    repeat {
      tries <- tries + 1L
      path <- NULL; gc(FALSE)
      tm <- timed(path <- run_path(nl_r))
      total_el <- total_el + tm[["elapsed"]]
      pinfo <- path_info(path$LM, nl_r)
      if (!any(pinfo$trunc) || tries > maxtries) break
      nl_r <- as.integer(ceiling(grow * nl_r))
    }
    add_timing(if (direct) "fastclime_direct" else "fastclime", r, tm, if (any(pinfo$trunc)) "truncated" else "ok",
               sprintf("%s;lambda.min=%.6g;nlambda=%d;tries=%d;elapsed_all_tries=%.3f;maxnlambda=%d;pivots_mean=%.1f;pivots_max=%d;trunc_cols=%d;infeasible_cols=%d",
                       base_details, lmin, nl_r, tries, total_el, path$maxnl, mean(pinfo$L) - 1, max(pinfo$L) - 1, sum(pinfo$trunc), sum(pinfo$infeas)))
    if (!direct && opt("fc_conly", TRUE, as.logical)) {   # bare C solver, same arguments as inside fastclime()
      Sd <- as.double(S); mu0 <- double(nl_r * p); ic0 <- double(as.numeric(nl_r) * p * p)   # allocated outside the timer
      tmc <- timed(str <- .C("parametric", Sd, as.integer(p), mu0, as.double(lmin), as.integer(nl_r),
                             as.integer(0), ic0, PACKAGE = "fastclime"))
      rm(ic0, str); gc(FALSE)
      add_timing("fastclime_Conly", r, tmc, "ok", sprintf("%s;nlambda=%d", base_details, nl_r))
    }
  }
  # exact solution at each grid lambda: the path is linear in lambda between consecutive breakpoints
  tm_int <- system.time({
    col_list <- lapply(seq_len(K), function(k) matrix(NaN, p, p))
    for (j in seq_len(p)) {
      Lj <- pinfo$L[j]; mus <- path$LM[seq_len(Lj), j]
      for (k in seq_len(K)) {
        lam <- lambda[k]
        if (lam >= mus[1]) { b <- path$sol(1L, j)
        } else {
          idx <- which(mus <= lam)
          if (!length(idx)) next                     # path ended above lam (truncated or infeasible)
          m <- idx[1]; hi <- mus[m - 1]; lo <- mus[m]
          w <- if (hi > lo) (lam - lo) / (hi - lo) else 0
          b <- (1 - w) * path$sol(m, j) + w * path$sol(m - 1L, j)
        }
        col_list[[k]][, j] <- b
      }
    }
    sym_list <- lapply(col_list, symmetrize_min)
  })
  add_timing("fastclime_interp", 1L, tm_int, "ok", "untimed-postprocessing:interpolate breakpoints to grid")
  # the package's selector returns the path solution at the first breakpoint <= lambda (not at lambda).
  # Replica of fastclime.selector() (same rule), usable in direct mode; checked against the real one.
  selector_icov <- function(lam) {
    icov <- matrix(0, p, p)
    for (i in seq_len(p)) {
      sq <- length(which(path$LM[, i] > lam))
      icov[, i] <- if (sq + 1 > path$maxnl) path$sol(sq, i) else path$sol(sq + 1L, i)
    }
    symmetrize_min(icov)
  }
  sel_check <- NA
  for (k in seq_len(K)) {
    sel <- selector_icov(lambda[k])
    if (!direct) {
      quiet(sel0 <- fastclime.selector(path$fit$lambdamtx, path$fit$icovlist, lambda[k]))
      sel_check <- max(sel_check, max(abs(sel0$icov - sel)), na.rm = TRUE)
    }
    if (want_write) write_bin(sel, sprintf("%s_selector_lam%d.bin", prefix, k))
  }
  if (!is.na(sel_check)) add_timing("fastclime_selector_check", 1L, NULL, "ok", sprintf("max|replica-fastclime.selector|=%.3g", sel_check))
  bp <- data.frame(col = rep(seq_len(p), pinfo$L), k = unlist(lapply(pinfo$L, seq_len)),
                   mu = unlist(lapply(seq_len(p), function(j) path$LM[seq_len(pinfo$L[j]), j])))
  write.csv(bp, paste0(prefix, "_breakpoints.csv"), row.names = FALSE)
}

## ---- flare -------------------------------------------------------------------------------------
if (method == "flare") {
  suppressPackageStartupMessages(library(flare))
  prec <- opt("flare_prec", 1e-4, as.numeric)
  maxite <- opt("flare_maxite", 1e4, as.numeric)
  rho <- opt("flare_rho", 1, as.numeric)
  for (r in seq_len(reps)) {
    tm <- timed(fit <- sugm(S, lambda = lambda, method = "clime", standardize = FALSE, perturb = FALSE,
                            prec = prec, max.ite = maxite, rho = rho, verbose = FALSE))
    if (!isTRUE(fit$cov.input)) stop("flare did not recognise the input as a covariance matrix")
    ite_full <- fit$ite[[3]]; ite1 <- fit$ite[[1]]; ite2 <- fit$ite[[2]]
    add_timing("flare", r, tm, "ok",
               sprintf("%s;prec=%g;max.ite=%g;rho=%g;perturb=%s;ite_stage1_sum=%d;ite_stage2_sum=%d;ite_full_sum=%d;stage1_hits_max=%d;stage2_hits_max=%d;full_hits_max=%d",
                       base_details, prec, maxite, rho, format(fit$perturb), sum(ite1), sum(ite2), sum(ite_full),
                       sum(ite1 >= maxite), sum(ite2 >= ceiling(maxite / 5)), sum(ite_full >= ceiling(maxite / 10))))
  }
  sym_list <- lapply(fit$icov, as.matrix)
  col_list <- lapply(fit$icov1, as.matrix)
}

## ---- clime ---------------------------------------------------------------------------------------
if (method %in% c("clime", "clime_simplex")) {
  suppressPackageStartupMessages(library(clime))
  linsolver <- if (method == "clime") "primaldual" else "simplex"
  pdtol <- opt("pdtol", 1e-3, as.numeric)
  pdmaxiter <- opt("pdmaxiter", 50L, as.integer)
  mode <- opt("clime_mode", "api")
  det <- sprintf("%s;linsolver=%s;pdtol=%g;pdmaxiter=%d;perturb=FALSE;mode=%s", base_details, linsolver, pdtol, pdmaxiter, mode)
  linprogS <- getFromNamespace("linprogS", "clime")
  linprogPD <- getFromNamespace("linprogPD", "clime")
  solve_cols <- function() {     # verbatim copy of the solve loop inside clime::clime(), keeping the columns
    Sigma <- S
    eigvals <- eigen(Sigma, only.values = TRUE)$values   # clime() always computes this (used by perturb)
    emat <- diag(p)
    out <- vector("list", K)
    if (linsolver == "simplex") {
      for (jl in 1:K) { Omega <- matrix(0, p, p)
        for (j in 1:p) Omega[, j] <- linprogS(Sigma, emat[, j], lambda[jl])
        out[[jl]] <- Omega }
    } else {
      Omega0 <- solve(Sigma)
      for (jl in 1:K) { Omega <- matrix(0, p, p)
        for (j in 1:p) Omega[, j] <- linprogPD(Omega0[, j], Sigma, emat[, j], lambda[jl], pdtol, pdmaxiter)
        out[[jl]] <- Omega }
    }
    out
  }
  ok <- TRUE
  for (r in seq_len(reps)) {
    res <- tryCatch({
      if (mode == "api") {
        tm <- timed(fit <- clime(S, lambda = lambda, sigma = TRUE, perturb = FALSE, standardize = FALSE,
                                 linsolver = linsolver, pdtol = pdtol, pdmaxiter = pdmaxiter))
        sym_list <- fit$Omegalist
      } else {
        tm <- timed(col_list <- solve_cols())
        sym_list <- lapply(col_list, symmetrize_min)
      }
      add_timing(method, r, tm, "ok", det); "ok"
    }, error = function(e) { add_timing(method, r, NULL, "error", paste(det, conditionMessage(e), sep = ";")); conditionMessage(e) })
    if (!identical(res, "ok")) { ok <- FALSE; break }
  }
  if (ok && mode == "api" && want_cols) {
    tm2 <- tryCatch(timed(col_list <- solve_cols()), error = function(e) NULL)
    if (!is.null(tm2)) {
      d <- max(sapply(seq_len(K), function(k) max(abs(symmetrize_min(col_list[[k]]) - sym_list[[k]]))))
      add_timing(paste0(method, "_colsolve"), 1L, tm2, "ok", sprintf("%s;max|sym(cols)-clime()|=%.3g", det, d))
    } else col_list <- NULL
  }
  if (!ok) { flush_timing(); cat("solver error:", res, "\n"); quit(status = 3) }
}

## ---- outputs -------------------------------------------------------------------------------------
for (k in seq_len(K)) {
  if (!want_write) break
  write_bin(sym_list[[k]], sprintf("%s_lam%d.bin", prefix, k))
  if (want_cols && !is.null(col_list)) write_bin(col_list[[k]], sprintf("%s_cols_lam%d.bin", prefix, k))
}
if (want_diag && !is.null(col_list)) {
  dl <- lapply(seq_len(K), function(k) {
    B <- col_list[[k]]
    R <- S %*% B - diag(p)
    maxres <- apply(abs(R), 2, max)
    data.frame(method = method, k = k, lambda = lambda[k], col = seq_len(p),
               l1 = colSums(abs(B)), maxres = maxres, viol_rel = (maxres - lambda[k]) / lambda[k],
               feasible = maxres <= lambda[k] * (1 + 1e-6), nnz = colSums(abs(B) > 1e-8))
  })
  dd <- do.call(rbind, dl)
  write.csv(dd, paste0(prefix, "_diag.csv"), row.names = FALSE)
  agg <- do.call(rbind, lapply(split(dd, dd$k), function(x) data.frame(
    k = x$k[1], lambda = x$lambda[1], sum_l1 = sum(x$l1), max_viol_rel = max(x$viol_rel, na.rm = TRUE),
    n_infeasible = sum(!x$feasible, na.rm = TRUE), n_nan = sum(is.na(x$l1)), mean_nnz = mean(x$nnz))))
  print(agg, digits = 6, row.names = FALSE)
}
flush_timing()
print(do.call(rbind, timing_rows)[, c("method", "p", "K", "rep", "elapsed", "status")], row.names = FALSE)
