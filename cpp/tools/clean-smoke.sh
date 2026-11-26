#!/usr/bin/env bash
set -euo pipefail
# Arguments: unpacked Ubuntu base filesystem, release tar.gz, new results directory.
base=$(realpath "$1")
archive=$(realpath "$2")
mkdir "$3"
results=$(realpath "$3")
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
tar -xf "$archive" -C "$stage"
root=("$stage"/*)
bwrap --unshare-all --die-with-parent --clearenv --setenv PATH /usr/bin:/bin \
  --setenv OPENBLAS_NUM_THREADS 1 --ro-bind "$base" / --proc /proc --dev /dev \
  --ro-bind "${root[0]}" /opt --bind "$results" /mnt /bin/sh -ec '
    test ! -e /usr/bin/R && ! command -v Rscript
    cd /opt
    cat /etc/os-release
    getconf GNU_LIBC_VERSION
    ldd bin/fsblup
    bin/fsblup build --ped example/ped.tsv --snp example/snp.tsv \
      --omics example/omics.tsv --out /mnt/kernels
    bin/fsblup fit --A /mnt/kernels/A.tsv --G /mnt/kernels/G.tsv \
      --O /mnt/kernels/O.tsv --pheno example/pheno.tsv --id ID --trait trait \
      --covar age --factor herd --alpha .2 --beta .5 --out /mnt/fit
    bin/fsblup fit --A example/A.tsv --G example/G.tsv --O example/O.tsv \
      --pheno example/pheno.tsv --id ID --trait trait --folds-file example/folds.tsv \
      --metric rmse --out /mnt/tuned
    awk -F "\t" '\''NR==FNR {if(FNR>1){u[$1]=$2;y[$1]=$3};next}
      FNR>1 {if(!($1 in u) || ($3-u[$1])^2>1e-10 || ($4-y[$1])^2>1e-10) exit 1;
      n++; if($2=="candidate")c++} END {if(n!=24 || c!=4)exit 1}'\'' \
      example/reml.tsv /mnt/fit/predictions.tsv
    # Eigenvalues of the shifted two-ID example are 3+1e-6 and -1+1e-6.
    printf "ID\t001\t002\n001\t1\t2\n002\t2\t1\n" > /mnt/indefinite.tsv
    bin/fsblup fuse --A /mnt/indefinite.tsv --G /mnt/indefinite.tsv --O /mnt/indefinite.tsv \
      --alpha 0 --beta 0 --pd-policy nearPD --out /mnt/projected
    awk -F "\t" '\''NR==2 && $1=="K" && $3=="nearPD" && $5==2 && $6==1 && $8>0 {ok=1}
       END {exit !ok}'\'' /mnt/projected/pd.tsv
    awk -F "\t" '\''BEGIN {l=3.000001; d=l/2; o=d*(1-1e-8)/(1+1e-8)}
      NR==2 {ok1=(($2-d)^2+($3-o)^2 < 1e-22)}
      NR==3 {ok2=(($3-d)^2+($2-o)^2 < 1e-22)}
      END {exit !(NR==3 && ok1 && ok2)}'\'' /mnt/projected/K.tsv
    cat /mnt/projected/pd.tsv
    cat /mnt/fit/model.tsv /mnt/tuned/tuning.tsv
  ' | tee "$results/smoke.log"
