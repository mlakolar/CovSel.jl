# Optional independent numerical reference using Patrick Danaher's published
# JGL 2.3.2 implementation. No R packages are required: only its ADMM and four
# base-R helpers are sourced, bypassing its igraph screening wrapper.
#
# Rscript examples/danaher2014/compare_jgl.R OUTPUT_DIRECTORY
#
# The Julia run must have --export-data=true. The first invocation downloads
# the MIT-licensed source files into tempdir(), pinned to the CRAN release.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1)
directory <- args[1]
commit <- "52fc1717ad075c0478330b029d005eaaf1f52690"
files <- c("admm.iters.r", "flsa.general.R", "flsa2.R", "soft.R", "penalty.as.matrix.R")
for (file in files) {
    url <- paste0("https://raw.githubusercontent.com/cran/JGL/", commit, "/R/", file)
    dest <- file.path(tempdir(), paste0("JGL-", file))
    download.file(url, dest, quiet=TRUE, mode="wb")
    source(dest)
}
Y <- lapply(1:3, function(k) as.matrix(read.table(file.path(directory, paste0("data_group_", k, ".tsv")))))
rows <- read.delim(file.path(directory, "replicates.tsv"))
rows <- rows[rows$rep == 1, ]
S <- lapply(Y, function(y) cov(y) * (nrow(y)-1)/nrow(y))
objective <- function(theta, lambda1, lambda2) {
    sum(sapply(1:3, function(k) sum(S[[k]]*theta[[k]]) - as.numeric(determinant(theta[[k]], logarithm=TRUE)$modulus))) +
        lambda1 * sum(sapply(theta, function(a) sum(abs(a)) - sum(abs(diag(a))))) +
        lambda2 * sum(sapply(list(c(1,2), c(1,3), c(2,3)), function(pair) sum(abs(theta[[pair[1]]]-theta[[pair[2]]]))))
}
checks <- list()
for (i in seq_len(nrow(rows))) {
    l1 <- rows$lambda1[i]
    l2 <- rows$lambda2[i]
    julia <- lapply(1:3, function(k) {
        path <- file.path(directory, paste0("fit_", as.character(l1), "_", as.character(l2), "_group_", k, ".tsv"))
        # Julia prints 0.0 where R prints 0; find the matching numeric prefix.
        if (!file.exists(path)) {
            candidates <- list.files(directory, pattern=paste0("_group_", k, "\\.tsv$"), full.names=TRUE)
            candidates <- candidates[grepl("/fit_", candidates)]
            path <- candidates[sapply(candidates, function(f) {
                fields <- strsplit(sub("^fit_", "", basename(f)), "_")[[1]]
                as.numeric(fields[1]) == l1 && as.numeric(fields[2]) == l2
            })]
        }
        stopifnot(length(path) == 1)
        as.matrix(read.table(path))
    })
    ref <- admm.iters(Y, l1, l2, penalty="fused", rho=1, weights=rep(1, 3),
                      penalize.diagonal=FALSE, maxiter=5000, tol=1e-5)
    difference <- max(sapply(1:3, function(k) max(abs(julia[[k]] - ref$Z[[k]]))))
    obj_julia <- objective(julia, l1, l2)
    obj_r <- objective(ref$Z, l1, l2)
    relative_objective <- abs(obj_julia - obj_r)/max(1, abs(obj_r))
    checks[[i]] <- data.frame(lambda1=l1, lambda2=l2, max_abs_difference=difference,
                              relative_objective_difference=relative_objective, r_iterations=ref$iters)
    cat(sprintf("lambda1=%.3f lambda2=%.3f max difference=%.3g relative objective=%.3g\n", l1, l2, difference, relative_objective))
    # The R all-pairs implementation merges near ties at 1e-4; it is not an
    # exact-prox oracle below that tolerance. Check numerical agreement on the
    # same objective, allowing for that deliberate R approximation.
    stopifnot(ref$iters < 5000, difference < 1e-3, relative_objective < 1e-6)
}
write.table(do.call(rbind, checks), file.path(directory, "jgl_reference.tsv"),
            sep="\t", row.names=FALSE, quote=FALSE)
cat("Reference: JGL 2.3.2, CRAN mirror commit", commit, "\n")
