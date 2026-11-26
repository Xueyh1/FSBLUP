# Run from cpp, using the R interpreter and packages in versions.tsv.
stopifnot(getRversion()=="4.4.1",packageVersion("AGHmatrix")=="2.1.4",
          packageVersion("rrBLUP")=="4.6.3",packageVersion("Matrix")=="1.7-0")
suppressPackageStartupMessages({library(dplyr);library(rrBLUP)})
source("../R/FSBLUP.R")
options(digits=17)
d <- "tests/data"
put <- function(df,name) write.table(df,file.path(d,paste0(name,".tsv")),
    sep="\t",quote=FALSE,row.names=FALSE)
matrixfile <- function(x,name) put(data.frame(ID=rownames(x),x,check.names=FALSE),name)
put(data.frame(package=c("R","rrBLUP","Matrix","AGHmatrix"),
 version=c(as.character(getRversion()),sapply(c("rrBLUP","Matrix","AGHmatrix"),
 function(p) as.character(packageVersion(p))))),"versions")
set.seed(135)
n <- 24; id <- sprintf("%03d",seq_len(n)); t <- 1:20; s <- 9:24
ped <- data.frame(ID=id,Sire=c(rep("0",8),id[rep(1:4,4)]),
 Dam=c(rep("0",8),id[rep(5:8,each=4)]))
put(ped,"ped")
A <- AGHmatrix::Amatrix(ped,ploidy=2)[id,id]
snp <- matrix(sample(0:2,n*17,replace=TRUE),n,dimnames=list(id,paste0("m",1:17)))
snp[1:3,1] <- -9;snp[1:12,2] <- NA;snp[1:13,3] <- -9
matrixfile(snp,"snp"); G <- AGHmatrix::Gmatrix(snp)[id,id]
om <- matrix(rnorm(n*9)+2,n,dimnames=list(id,paste0("o",1:9)));om[1,1]<-NA
matrixfile(om,"omics");O <- FSBLUP.Omatrix(om)
for (name in c("A","G","O")) matrixfile(get(name),name)
combine <- function(a,b,A=A0,G=G0,O=O0) FSBLUP.CombineMat(NULL,A,G,O,a,b)
A0<-A;G0<-G;O0<-O
a <- .2;b <- .5
K <- combine(a,b); matrixfile(K,"K");matrixfile((1-a-b)*A+b*G+a*O,"K_raw")
for (pair in list(c(0,0),c(0,1),c(1,0))) matrixfile(combine(pair[1],pair[2]),paste0("end",paste(pair,collapse="")))
# BB000 independent intermediates, plus the baseline's actual final return.
Ob<-O[s,s]+diag(length(s));matrixfile(Ob,"Ob");base<-adj_pos(G[s,s]);reg<-G[-s,s]%*%solve(base)
sw<-(mean(base)-mean(diag(base)))/(mean(Ob)-mean(diag(Ob)))
sm<-mean(diag(Ob))-sw*mean(diag(base))
cal<-sw*Ob+sm;replacement<-a*cal+(1-a)*base
H<-G;H[-s,-s]<-G[-s,-s]+reg%*%(replacement-base)%*%t(reg)
H[-s,s]<-reg%*%replacement;H[s,-s]<-t(H[-s,s]);H[s,s]<-replacement
matrixfile(base,"B_base");matrixfile(cal,"B_cal");matrixfile(replacement,"B_T");matrixfile(H,"B_H")
Kb<-combine(a,b,O=Ob)[id,id];matrixfile(Kb,"B_K")
stopifnot(max(abs(Kb-adj_pos(b*H/mean(diag(H))+(1-b)*A)))<1e-8)
matrixfile(combine(a,b,A[t,t],G[t,t],Ob[id[intersect(t,s)],id[intersect(t,s)]])[id[t],id[t]],"B_Kref")
# A 6-animal example, groups 1={1,2}, 2={3,4}, 4={5,6}. Explicit
# individual covariances implement the displayed entries of equations 3/7.
i<-id[1:6]; ca<-outer(1:6,1:6,function(x,y).28^abs(x-y));dimnames(ca)<-list(i,i)
cg<-.85*ca[3:6,3:6]+.15*diag(4);co<-matrix(c(1.2,.35,.35,.9),2,dimnames=list(i[5:6],i[5:6]))
co2<-co;co2[1,1]<-1.4;co2[1,2]<-co2[2,1]<-.12
matrixfile(ca,"C_A");matrixfile(cg,"C_G");matrixfile(co,"C_O");matrixfile(co2,"C_perturb")
nested <- function(m,a,b){
 g<-3:4;s<-5:6; c<-a*m+b*cg[3:4,3:4]+(1-a-b)*ca[s,s];res<-ca
 for (j in 1:6) for (k in j:6) {
 v <- if(k<=2) ca[j,k]+ca[j,s]%*%solve(ca[s,s],(c-ca[s,s])%*%solve(ca[s,s],ca[s,k])) else
 if(j<=2&&k<=4) ca[j,g]%*%solve(ca[g,g],cg[1:2,k-2]) else
 if(k<=4) cg[j-2,k-2] else if(j<=4) ca[j,s]%*%solve(ca[s,s],c[,k-4]) else c[j-4,k-4]
 res[j,k]<-res[k,j]<-as.numeric(v)
 };res
}
for (name in c("C_expected","C_changed","C_zero")) {
 val<-switch(name,C_expected=nested(co,a,b),C_changed=nested(co2,a,b),C_zero=nested(co,0,b))
 matrixfile(val,name)
}
legacy<-combine(a,b,ca,cg,co);stopifnot(identical(legacy,combine(a,b,ca,cg,co2)))
matrixfile(legacy,"C_legacy")
# Select a deterministic, identifiable interior REML example.
x<-cbind(intercept=1,age=rep(1:6,4),herdB=rep(0:1,12))
for(seed in 1:100){set.seed(seed);y<-drop(t(chol(K+.6*diag(n)))%*%rnorm(n)+x%*%c(.3,.08,.2))
 y[-t]<-NA; fit<-mixed.solve(y,X=x,K=K);if(fit$Ve/fit$Vu>.2&&fit$Ve/fit$Vu<2)break}
put(data.frame(ID=id,trait=y,age=x[,2],herd=ifelse(x[,3]==1,"B","A"),year=rep(2020:2023,each=6),
 binary=rep(0:1,12)),"pheno")
put(data.frame(id=id,u=fit$u,yhat=drop(x%*%fit$beta)+fit$u),"reml")
put(data.frame(term=c("Vu","Ve","lambda",colnames(x)),value=c(fit$Vu,fit$Ve,fit$Ve/fit$Vu,fit$beta)),"parameters")
v<-K[t,t]+.7*diag(length(t));bhat<-solve(crossprod(x[t,],solve(v,x[t,])),crossprod(x[t,],solve(v,y[t])))
u<-K[,t]%*%solve(v,y[t]-x[t,]%*%bhat)
put(data.frame(id=id,u=drop(u),yhat=drop(x%*%bhat+u)),"fixed_lambda")
put(data.frame(term=colnames(x),value=drop(bhat)),"fixed_lambda_b")
fold<-rep(1:4,length.out=length(t));put(data.frame(`repeat`=1,fold=fold,id=id[t],check.names=FALSE),"folds")
scores<-list()
for (ai in 0:4) for (bi in 0:(4-ai)){
 k<-combine(ai/4,bi/4,A[t,t],G[t,t],O[t,t]);v<-sapply(1:4,function(f){
 yy<-y[t];yy[fold==f]<-NA;pred<-mixed.solve(yy,K=k)$u[fold==f];sqrt(mean((y[t][fold==f]-pred)^2))})
 scores[[length(scores)+1]]<-c(ai/4,bi/4,v)
};put(as.data.frame(do.call(rbind,scores)),"grid_scores")
# Evaluate the literal tryCatch idiom in the R source with instrumentation.
calls<-list();record<-function(y,K){calls[[length(calls)+1]]<<-K;mixed.solve(y,K=K)}
kk<-K;old<-tryCatch(record(y,K=kk),error={kk=adj_pos(kk);record(y,K=kk)})
stopifnot(length(calls)==2,identical(calls[[1]],calls[[2]]));matrixfile(kk,"legacy_solver_K")
put(data.frame(id=id,u=old$u),"legacy_u")
# Independent reference for the greatest rrBLUP fold discrepancy. Evaluate a
# Cholesky REML profile directly, with numerical optimization on log(lambda).
ai<-.5;bi<-0; f<-2
kf<-combine(ai,bi,A[t,t],G[t,t],O[t,t]);train<-fold!=f;valid<-!train
xx<-matrix(1,sum(train),1);yy<-y[t][train];df<-length(yy)-1
direct<-function(loglam,details=FALSE){
 lam<-exp(loglam);kt<-kf[train,train]+lam*diag(sum(train))
 L<-t(chol(kt));wx<-forwardsolve(L,xx);wy<-forwardsolve(L,yy)
 b<-sum(wx*wy)/sum(wx^2);res<-wy-wx*b
 obj<-df*log(sum(res^2)/df)+2*sum(log(diag(L)))+log(sum(wx^2))-log(sum(xx^2))
 pred<-kf[valid,train]%*%backsolve(t(L),res)
 if(details)c(lambda=lam,objective=obj,rmse=sqrt(mean((y[t][valid]-pred)^2))) else obj
}
bounds<-log(c(1e-9,1e9));best<-optimize(direct,bounds,tol=1e-12)$minimum
points<-c(bounds,best);hp<-direct(points[which.min(sapply(points,direct))],TRUE)
masked<-y[t];masked[valid]<-NA; rr<-mixed.solve(masked,K=kf)
rrval<-direct(log(rr$Ve/rr$Vu),TRUE)
put(data.frame(method=c("direct","rrBLUP"),rbind(hp,rrval),row.names=NULL),"boundary")
put(data.frame(method="mixed.solve",LL=fit$LL,profile_objective=
 -2*fit$LL-(length(t)-ncol(x))*(1+log(2*3.14159))),"likelihood")
# Run the baseline entry point, wrapping the scoring callback to retain its
# whole-square search trajectory. Its native folds are intentionally distinct.
phe<-data.frame(id=id,trait=y)
original_cv<-cv.cal_ab
trajectory<-list()
cv.cal_ab<-function(...){
 value<-original_cv(...)
 trajectory[[length(trajectory)+1L]]<<-as.data.frame(value)
 value
}
legacy_prediction<-FSBLUP(phe,M1=A,M2=G,M3=O,trait_col="trait",
 po.crv.num=4,po.crv.rep.num=1,bi_max_iter=2)
put(do.call(rbind,trajectory),"legacy_search")
put(data.frame(id=id,u=legacy_prediction),"legacy_search_u")
