# Rebuild the offline nearPD fixture with Rscript --vanilla nearpd_generate.R nearpd_cases.tsv
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
stopifnot(as.character(utils::packageVersion("Matrix")) == "1.7.0")
ref <- function(x, ...) {
    suppressWarnings(Matrix::nearPD(x, base.matrix = TRUE, ...))
}
output <- file(args[1L], open = "w")
writeLines("# Matrix 1.7-0. One case per line: name n corr keepDiag maxit eig.tol conv.tol posd.tol iter converged rel.tol normF, followed by input and result in row-major order.", output)
record <- function(name, x, corr = FALSE, keepDiag = FALSE, maxit = 100L,
                   eig.tol = 1e-6, conv.tol = 1e-7, posd.tol = 1e-8) {
    model <- ref(x, corr = corr, keepDiag = keepDiag, maxit = maxit,
                 eig.tol = eig.tol, conv.tol = conv.tol, posd.tol = posd.tol)
    values <- c(nrow(x), corr, keepDiag, maxit, eig.tol, conv.tol, posd.tol,
                model$iterations, model$converged, model$rel.tol, model$normF,
                as.vector(t(x)), as.vector(t(model$mat)))
    writeLines(paste(c(name, sprintf("%.17g", values)), collapse = "\t"), output)
}
a <- matrix(c(2, 1.7, -.9, 1.7, 1, 1.2, -.9, 1.2, .7), 3, 3, byrow = TRUE)
record("already_pd", matrix(c(1.1, .18, .18, .8), 2))
record("indefinite", a)
record("unequal_diagonal", a, keepDiag = TRUE)
record("correlation", a, corr = TRUE)
record("corr_priority", a, corr = TRUE, keepDiag = TRUE)
record("one_iteration", a, keepDiag = TRUE, maxit = 1L)
record("custom_tolerances", a, corr = TRUE, posd.tol = 1e-5, conv.tol = 1e-9,
       eig.tol = 1e-7)
record("rank_one", tcrossprod(c(1, 2, -.5)))
record("spectral_cutoff", diag(c(2, 3e-6, 1e-6)))
record("asymmetric", matrix(c(1, 2.2, 2.0, 1), 2, 2, byrow = TRUE))
record("scalar", matrix(2.5, 1, 1))
record("scale", 2.5e5 * a)
# The R baseline must reject zero and all-negative spectra.
for (x in list(matrix(0,2,2), -diag(3)))
    stopifnot(inherits(try(ref(x), silent = TRUE), "try-error"))
close(output)
