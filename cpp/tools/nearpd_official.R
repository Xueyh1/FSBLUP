# Rscript --vanilla nearpd_official.R EXTRACTED_OFFICIAL_EXAMPLE NEW_OUTPUT
# Reconstruct the documented C++ mode-C block equations in R, then run CRAN
# Matrix::nearPD on the jittered complete kernel and rrBLUP on the age submatrix.
# This comparison isolates the C++ repair and fit; upstream R mode C has
# different raw-fusion equations.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L, as.character(packageVersion("Matrix")) == "1.7.0",
          as.character(packageVersion("rrBLUP")) == "4.6.3")
root <- args[1]; out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
read_kernel <- function(k) {
    x <- read.csv(file.path(root, paste0("test_mat",k,".csv")), row.names=1,
                  check.names=FALSE)
    as.matrix(x)[,rownames(x),drop=FALSE]
}
A <- read_kernel("A"); G <- read_kernel("G"); O <- read_kernel("O")
s <- rownames(O); g <- setdiff(rownames(G), s); u <- setdiff(rownames(A), rownames(G))
stopifnot(nrow(A)==1407L, nrow(G)==198L,nrow(O)==74L, all(s %in% rownames(G)),all(rownames(G) %in% rownames(A)))
A44 <- A[s,s]; C <- .2*O+.5*G[s,s]+.3*A44
R14 <- t(solve(A44,A[s,u])); R12 <- t(solve(A[g,g],A[g,u])); R24 <- t(solve(A44,A[s,g]))
K <- A
K[u,u] <- A[u,u] + R14 %*% (C-A44) %*% t(R14)
K[g,g] <- G[g,g]; K[s,s] <- C
K[u,g] <- R12 %*% G[g,g]; K[u,s] <- R14 %*% C; K[g,s] <- R24 %*% C
K[g,u] <- t(K[u,g]); K[s,u] <- t(K[u,s]); K[s,g] <- t(K[g,s])
K <- (K+t(K))/2
K0 <- K
diag(K) <- diag(K)+1e-6
t <- system.time(fit <- Matrix::nearPD(K))
stopifnot(fit$converged)
kp <- as.matrix(fit$mat)
raw_l <- eigen(K0,symmetric=TRUE,only.values=TRUE)$values
ev <- eigen(kp,symmetric=TRUE,only.values=TRUE)$values
cat("R=", as.character(getRversion()), " Matrix=", as.character(packageVersion("Matrix")), "\n",sep="")
cat(sprintf("n=%d raw_min=%.17g repaired_min=%.17g repaired_max=%.17g normF=%.17g relative_normF=%.17g max_abs_change=%.17g max_abs_diag=%.17g iter=%d converged=%s rel_tol=%.17g elapsed=%.4fs\n",
nrow(K),min(raw_l),min(ev),max(ev),fit$normF,fit$normF/norm(K,"F"),max(abs(kp-K)),max(abs(diag(kp)-diag(K))), fit$iterations,fit$converged,fit$rel.tol,unname(t[3])))
writeBin(as.double(K0),file.path(out,"raw.bin"),size=8,endian="little")
writeBin(as.double(kp),file.path(out,"nearPD.bin"),size=8,endian="little")
writeLines(rownames(K),file.path(out,"ids.txt"))
d <- read.csv(file.path(root,"test_pheno.csv"),colClasses=c(id="character"),check.names=FALSE)
stopifnot(nrow(d)==198L,!anyNA(d$age),all(d$id %in% rownames(kp)))
hold <- seq.int(1L,nrow(d),by=10L); write.table(d[hold,c("id","age")],file.path(out,"heldout.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
d$age[hold] <- NA_real_
options(digits=17)
write.table(d,file.path(out,"masked.tsv"),sep="\t",quote=FALSE,row.names=FALSE,na="NA")
m <- rrBLUP::mixed.solve(y=d$age,K=kp[d$id,d$id])
pred <- data.frame(id=d$id,u=as.double(m$u),yhat=as.double(m$u)+as.double(m$beta))
write.table(pred,file.path(out,"r_predictions.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
reml <- data.frame(Vu=m$Vu, Ve=m$Ve, lambda=m$Ve/m$Vu, intercept=as.double(m$beta))
write.table(reml,file.path(out,"r_model.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
cat(sprintf("R_fit_reference=%d candidate=%d Vu=%.17g Ve=%.17g lambda=%.17g b=%.17g max_abs_candidate_u=%.17g\n",
sum(!is.na(d$age)),sum(is.na(d$age)),m$Vu,m$Ve,m$Ve/m$Vu,m$beta,max(abs(pred$u[hold]))))
