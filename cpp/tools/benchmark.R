# Rscript --vanilla tools/benchmark.R unpacked-example fresh-output (run from cpp)
args<-commandArgs(TRUE);input<-args[1];out<-args[2];dir.create(out,recursive=TRUE)
suppressPackageStartupMessages({library(dplyr);library(rrBLUP)})
source("../R/FSBLUP.R")
read<-function(file)read.csv(file.path(input,paste0("test_",file,".csv")),check.names=FALSE)
readmat<-function(k){df<-read(paste0("mat",k));rownames(df)<-df$ID;as.matrix(df[,-1])}
O<-readmat("O");ids<-rownames(O);G<-readmat("G")[ids,ids];A<-readmat("A")[ids,ids]
ph<-read("pheno");ph<-ph[match(ids,ph$id),];stopifnot(identical(as.character(ph$id),ids))
ph$age[seq(1,length(ids),by=10)]<-NA
write.table(ph,file.path(out,"pheno.tsv"),row.names=FALSE,quote=FALSE,sep="\t")
for(k in c("A","G","O"))write.table(data.frame(ID=ids,get(k),check.names=FALSE),
 file.path(out,paste0(k,".tsv")),row.names=FALSE,quote=FALSE,sep="\t")
# Benchmark exactly the files consumed by the C++ fixed-weight run.
read_bench<-function(name){df<-read.delim(file.path(out,paste0(name,".tsv")),check.names=FALSE)
 rownames(df)<-df$ID;as.matrix(df[,-1])}
elapsed<-function()as.numeric(proc.time()["elapsed"])
st<-elapsed();A<-read_bench("A");G<-read_bench("G");O<-read_bench("O")
ph<-read.delim(file.path(out,"pheno.tsv"));t1<-elapsed()
k<-FSBLUP.CombineMat(NULL,A,G,O,.2,.5);t2<-elapsed()
m<-mixed.solve(ph$age,K=k);t3<-elapsed()
write.table(data.frame(id=ids,u=m$u),file.path(out,"r-u.tsv"),row.names=FALSE,quote=FALSE,sep="\t")
write.table(data.frame(read=t1-st,fuse=t2-t1,solve=t3-t2,total=elapsed()-st,BLAS=La_library()),
 file.path(out,"r-timing.tsv"),row.names=FALSE,quote=FALSE,sep="\t")
