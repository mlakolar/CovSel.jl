# Install the R implementations of CLIME compared in README.md into a library:
#
#   Rscript install_r.R <library>
#
# fastclime was archived on CRAN in 2023 and is installed from the archive.
args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args)) args[1] else "Rlib"
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
repos <- "https://cloud.r-project.org"
need <- c("lpSolve", "igraph", "Matrix", "flare", "clime")
missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = lib, repos = repos)
if (!requireNamespace("fastclime", quietly = TRUE))
  install.packages(paste0(repos, "/src/contrib/Archive/fastclime/fastclime_1.4.1.1.tar.gz"),
                   lib = lib, repos = NULL, type = "source")
for (pkg in c(need, "fastclime")) cat(pkg, as.character(packageVersion(pkg)), "\n")
