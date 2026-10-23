# Time the precision matrix estimator of the R package scalreg (Sun, 2013) on the
# data written by speed.jl, at the penalty level of scaledlasso, and write its
# estimate for the comparison.
#
#   Rscript bench_r.R DIR N P OUT.csv [lib=PATH]
#
# DIR holds X.bin (N × P doubles, column-major) and lambda0.txt. scalreg 1.0.1 is
# archived on CRAN; install it from the archive (it needs lars), as in
# ../clime/install_r.R. Writes DIR's name, the seconds and the largest
# difference to est.bin (relative to its largest entry) to OUT.csv.

args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; n <- as.integer(args[2]); p <- as.integer(args[3]); out <- args[4]
lib <- sub("^lib=", "", grep("^lib=", args, value = TRUE))
if (length(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(scalreg))

X <- matrix(readBin(file.path(dir, "X.bin"), "double", n * p), n, p)
lam0 <- as.numeric(readLines(file.path(dir, "lambda0.txt")))
t <- system.time(est <- scalreg(X, lam0 = lam0))[["elapsed"]]
ref <- matrix(readBin(file.path(dir, "est.bin"), "double", p * p), p, p)
d <- max(abs(est$precision - ref)) / max(abs(ref))
cat(sprintf("%s,scalreg,%.6g,%d,%.3g\n", basename(dir), t, sum(est$precision != 0), d),
    file = out, append = TRUE)
