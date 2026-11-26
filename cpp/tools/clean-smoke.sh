#!/usr/bin/env bash
set -euo pipefail
# Arguments: minimal Linux root filesystem, release tar.gz, new results directory.
# Optionally append the official example directory and independently masked age TSV.
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
      --covar age --factor herd --alpha .2 --beta .5 --threads 2 --out /mnt/fit
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
    nested=source/cpp/tests/nested
    bin/fsblup fit --A "$nested/A.tsv" --G "$nested/G.tsv" --O "$nested/O.tsv" \
      --pheno "$nested/pheno.tsv" --id ID --trait trait --folds-file "$nested/folds.tsv" \
      --metric rmse --strict-simplex --pd-policy nearPD --parallel weights --threads 4 --out /mnt/outer
    awk -F "\t" '\''NR==2 && $2==0.75 && $3==0.25 {ok=1} END {exit !ok}'\'' /mnt/outer/fusion.tsv
    awk -F "\t" '\''NR==2 && $2==1 && $5==4 {ok=1} END {exit !ok}'\'' /mnt/outer/runtime.tsv
    bin/fsblup prepare --ped example/ped.tsv --snp example/snp.tsv --omics example/omics.tsv \
      --snp-format 2bit --out /mnt/features
    bin/fsblup fit --cache /mnt/features --pheno example/pheno.tsv --trait trait --id ID \
      --covar age --factor herd --alpha .2 --beta .5 --varcomp em-reml --trace exact --out /mnt/cache-em
    awk -F "\t" '\''NR==2 && ($1-0.604754055)^2<1e-10 && ($2-0.837449262)^2<1e-10 {ok=1} END {exit !ok}'\'' /mnt/cache-em/model.tsv
    sd=source/cpp/tests/pcg_data
    bin/fsblup fit --A "$sd/kernel.tsv" --G "$sd/kernel.tsv" --O "$sd/kernel.tsv" \
      --pheno "$sd/trait.tsv" --trait y --alpha 0 --beta 0 --trace hutchinson \
      --probe-solver lanczos --slq-reorth local --slq-batch 4 --slq-depth 24 \
      --probe-seed 19 --trace-rank 4 --probes 16 --max-probes 4096 --mc-tol .025 --out /mnt/random
    awk -F "\t" '\''NR==2 && $4=="approx-interior" && (log($3/.5))^2<.0025 {ok=1} END {exit !ok}'\'' /mnt/random/model.tsv
    cat /mnt/random/model.tsv
    cat /mnt/cache-em/model.tsv
    cat /mnt/outer/runtime.tsv /mnt/outer/fusion.tsv
    cat /mnt/projected/pd.tsv
    cat /mnt/fit/model.tsv /mnt/tuned/tuning.tsv
  ' | tee "$results/smoke.log"
if [[ $# == 5 ]]; then
  src=$(realpath "$4")
  cp "$5" "$results/masked.tsv"
  bwrap --unshare-all --die-with-parent --clearenv --setenv OPENBLAS_NUM_THREADS 1 \
    --ro-bind "$base" / --proc /proc --dev /dev --ro-bind "${root[0]}" /opt \
    --ro-bind "$src" /srv --bind "$results" /mnt /opt/bin/fsblup fit \
    --A /srv/test_matA.csv --G /srv/test_matG.csv --O /srv/test_matO.csv \
    --pheno /mnt/masked.tsv --id id --trait age --alpha .2 --beta .5 \
    --pd-policy nearPD --threads 1 --write-k --out /mnt/official
fi
