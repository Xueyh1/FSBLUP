# Rscript --vanilla tests/nearpd_nested.R OUTPUT_DIRECTORY
# Weight-sensitive mode-C cohort with all three groups among references.
# Use direct conditional-block equations and concentrated GLS likelihood.
stopifnot(packageVersion("Matrix") == "1.7-0")
out <- commandArgs(TRUE)[1]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
options(digits=17)
put <- function(x, file) write.table(x,file.path(out,paste0(file,".tsv")),
                                    row.names=FALSE,quote=FALSE,sep="\t")
kernel <- function(x,file) put(data.frame(ID=rownames(x),x,check.names=FALSE),file)
ids <- sprintf("%03d",1:30)
gids <- ids[9:30]; oids <- ids[18:30]
serial <- function(ids,rho) {
    m <- rho^abs(outer(seq_along(ids),seq_along(ids),"-"))
    dimnames(m) <- list(ids,ids)
    m
}
A <- serial(ids,.22); A[1,9] <- A[9,1] <- .86
G <- serial(gids,.48); G[1,1] <- 4
O <- serial(oids,-.52)
stopifnot(all(sapply(list(A,G,O),function(x) min(eigen(x,symmetric=TRUE)$values)>0)))
# Independently permute the file axes, including the G and O conditioning sets.
kernel(A[30:1,c(2:30,1)],"A")
kernel(G[c(2:22,1),22:1],"G")
kernel(O[13:1,c(4:13,1:3)],"O")
raw_kernel <- function(pop,a,b) {
    s <- intersect(pop,oids); g <- setdiff(intersect(pop,gids),s)
    u <- setdiff(pop,gids)
    common <- a*O[s,s]+b*G[s,s]+(1-a-b)*A[s,s]
    base <- A[pop,pop]
    base[u,u] <- A[u,u]+A[u,s]%*%solve(A[s,s],(common-A[s,s])%*%solve(A[s,s],A[s,u]))
    base[g,g] <- G[g,g]; base[s,s] <- common
    base[u,g] <- A[u,g]%*%solve(A[g,g],G[g,g])
    base[u,s] <- A[u,s]%*%solve(A[s,s],common)
    base[g,s] <- A[g,s]%*%solve(A[s,s],common)
    base[g,u]<-t(base[u,g]);base[s,u]<-t(base[u,s]);base[s,g]<-t(base[g,s])
    (base+t(base))/2
}
repaired <- function(raw) {
    S<-raw+diag(1e-6,nrow(raw)); p<-Matrix::nearPD(S)
    stopifnot(p$converged)
    as.matrix(p$mat)
}
set.seed(51)
i <- seq_along(ids)
y <- sin(.43*i)+cos(.18*i)+ifelse(ids %in% oids,2*(-1)^i,0)+.3*rnorm(length(i))
names(y)<-ids; y[ids[c(5,16,28)]]<-NA
pop<-ids[!is.na(y)]
fold<-setNames(rep(1:3,length.out=length(pop)),pop)
put(data.frame(ID=ids,trait=y),"pheno")
put(data.frame(`repeat`=1,fold=fold,id=pop,check.names=FALSE),"folds")
fit <- function(k,tr,va) {
    df<-length(tr)-1
    gls<-function(z,predict=FALSE) {
        V<-k[tr,tr]+exp(z)*diag(length(tr));L<-t(chol(V))
        wy<-forwardsolve(L,y[tr]);wx<-forwardsolve(L,rep(1,length(tr)))
        b<-sum(wx*wy)/sum(wx^2);r<-wy-wx*b
        if(predict) return(drop(k[va,tr,drop=FALSE]%*%backsolve(t(L),r)))
        df*log(sum(r^2)/df)+2*sum(log(diag(L)))+log(sum(wx^2))
    }
    z<-log(c(1e-9,1e9)); best<-optimize(gls,z,tol=1e-11)$minimum
    pts<-c(z,best);gls(pts[which.min(sapply(pts,gls))],TRUE)
}
scores<-function(a,b) {
    k<-repaired(raw_kernel(pop,a,b))
    v<-sapply(1:3,function(j){
        tr<-pop[fold!=j];va<-pop[fold==j]
        sqrt(mean((y[va]-fit(k,tr,va))^2))
    })
    c(alpha=a,beta=b,v,mean(v))
}
grid<-do.call(rbind,lapply(0:4,function(i)
    do.call(rbind,lapply(0:(4-i),function(j)scores(i/4,j/4)))))
colnames(grid)<-c("alpha","beta","r1f1","r1f2","r1f3","mean")
best<-grid[which.min(grid[,"mean"]),]; step<-.125
for(round in 1:10) {
    points<-expand.grid(beta=best["beta"]+c(-step,step),alpha=best["alpha"]+c(-step,step))
    points<-subset(points,alpha>=0 & beta>=0 & alpha+beta<=1)
    prev<-best["mean"]
    for(i in seq_len(nrow(points))){
        row<-scores(points$alpha[i],points$beta[i]);names(row)<-colnames(grid)
        grid<-rbind(grid,row)
        if(row["mean"] < best["mean"]-1e-12)best<-row
    }
    if(prev-best["mean"]<1e-4)break
    step<-step/2
}
put(as.data.frame(grid),"scores")
kernel(repaired(raw_kernel(ids,best["alpha"],best["beta"])),"selected_K")
