# fsblup 0.1.0

Linux 命令行工具，用于单性状、每个个体一条表型记录的融合核 BLUP。将候选个体保留在表型文件里，性状填 `NA`。在 `predictions.tsv` 中按 `role=candidate` 取其带 ID 的 `u` 和 `yhat`。未列入表型文件的系谱祖先可以参与构核。

## 直接运行

发布包适用于 x86_64 Linux、glibc 2.34 及以上。解包后保留 `bin/` 和 `lib/` 的相对位置。程序、预测与调参均可在无 R 环境运行。以下命令在解包目录执行，输出目录使用新名字。源码树中可分别设置 `exe=build/fsblup` 和 `d=tests/data`。

```bash
exe=bin/fsblup
d=example

# 从原始系谱、SNP 和组学开始。示例为连续型组学。
"$exe" build --ped "$d/ped.tsv" --snp "$d/snp.tsv" --omics "$d/omics.tsv" \
  --omics-type generic --out kernels
"$exe" fit --A kernels/A.tsv --G kernels/G.tsv --O kernels/O.tsv \
  --pheno "$d/pheno.tsv" --id ID --trait trait --covar age --factor herd \
  --alpha .2 --beta .5 --out fixed

# 从预构矩阵开始，按可重放折表选权重，再用全部参考个体拟合。
"$exe" fit --A "$d/A.tsv" --G "$d/G.tsv" --O "$d/O.tsv" \
  --pheno "$d/pheno.tsv" --id ID --trait trait --folds-file "$d/folds.tsv" \
  --metric rmse --out selected
awk -F '\t' 'NR==1 || $2=="candidate"' selected/predictions.tsv

# 只导出固定权重的融合核。
"$exe" fuse --A "$d/A.tsv" --G "$d/G.tsv" --O "$d/O.tsv" \
  --alpha .2 --beta .5 --out fused
```

## 输入与运行约定

- TSV 和 CSV 使用单行表头。方阵左上角可为空，第一列为行 ID；列 ID 要与行 ID 集合相同，两轴可独立换序。字符串 ID 区分大小写并保留前导零。数值必须有限。每个个体最多一行。
- 系谱使用 `ID,Sire,Dam`，忽略列名大小写。亲本可出现在后面。0 表示未知。存在已知 ID 但无系谱行的父母按无亲缘创始者加入。循环系谱会报错。
- 特征文件第一列为样本 ID，其余列为特征。SNP 支持二倍体 0/1/2、NA、空字段和 -9。组学缺失补零。FPKM 输入使用 `--omics-type fpkm`，连续型其它组学使用 `generic`。每层在其所有输入样本上计算标准化统计量。`qc.tsv` 记录保留特征、过滤、常数列和插补数，三层均输出 ID 清单。
- `fit` 默认含截距，`--covar age,weight` 指定数值列，`--factor herd,sex` 指定分类列。每个分类变量以字符串排序的第一水平为参照，写入 `reference_levels.tsv`。每次训练检查设计秩。候选个体也需提供固定效应值。
- `--alpha` 与 `--beta` 成对指定会跳过搜索。自动选择有三种方式：默认均衡随机折 `--folds 5 --repeats 2 --seed 42`、`--folds-file FILE`、`--year 2021` 或 `--ngen 3`。时间验证读取同名数值列，参考个体中小于等于截断值的用于训练，其余用于调参评分。随机参数、外部折表、时间截断互斥。`folds.tsv` 包含 repeat、fold、id，每次重复每个参考 ID 恰好一次。普通折从 1 连续编号；时间折中 0 表示训练，1 表示验证，可原样回读。
- 训练性状为 `--trait`，评分默认比较验证个体的 `u` 与该性状，可用 `--score-col` 指定另一个评分列。有表型个体的评分列须完整。Pearson 与二分类 AUC 取最大值，RMSE 取最小值。AUC 标签必须为 0/1，并列预测计半分；`u` 是随机效应预测量。调参分数使用相同可评分折的等权平均。`fold_counts.tsv` 解释仅由真值导致的不可评分折；预测恒定、求解失败等候选特有错误令整个候选失格。
- 调参先限制三层源矩阵到参考 ID，再融合。选定权重后在完整 ID 集合重新融合并用全部有表型个体重拟合。覆盖在参考集中退化时可指定固定权重。独立评估请将留出个体的拟合性状填为 NA，并在预测后单独评分。预构核可含所有样本的无监督特征统计量。
- `--threads N` 设置 OpenBLAS 并行，权重与折串行；默认 1。Slurm 上可用 `srun -n1 -c4 --mem=8G ... --threads 4`。`--memory-mb` 以 MiB 为单位，默认 8192，按 `(160 n² + 64 nm)/2²⁰` 估算密集工作集，n 为样本数，m 为输入特征数。估算与进程实际 RSS 分别写入 `memory.tsv` 和 `runtime.tsv`。Slurm 内存限额仍适用于整个进程。
- `fusion.tsv` 记录覆盖、权重和扰动。`fuse` 或 `fit --write-k` 输出完整 `K_raw.tsv`、`K.tsv` 和 ID。模型文件含 Vu、Ve、lambda、边界状态；`fixed.tsv` 含各固定效应。搜索输出 `search.tsv` 与 `tuning.tsv`。若所有网格点失格，检查保留的搜索轨迹和 `failure.txt`。运行结果以目录为单位发布。
- 支持 A/G/O 同集合、A=G 且 O 为真子集，以及 O 严格包含于 G、G 严格包含于 A 的嵌套覆盖。其它情形会给出分组数量和示例 ID。输入约定、分块公式和实测边界见 [数值验收记录](VALIDATION.md)。

## 从源码构建

需要 C++17、CMake 3.18+、Armadillo 头文件、LP64 OpenBLAS（含 LAPACK）。日常测试还需 Python 3，仅用标准库。已安装 Armadillo 的主机可省略 `-DARMADILLO_INCLUDE_DIR`。以下为本次 Slurm 主机上的完整命令，工作目录为仓库的 `cpp/`。本地头文件取自 Debian 的独立开发包。

```bash
mkdir -p .deps
curl -fL https://deb.debian.org/debian/pool/main/a/armadillo/libarmadillo-dev_11.4.2+dfsg-1_amd64.deb -o .deps/arma.deb
echo '5ca7da15ae30020a9d3e5a27c6dd14d2d73dd2ff3f81e50ecd2ff39e9a5c61a2  .deps/arma.deb' | sha256sum -c -
ar p .deps/arma.deb data.tar.xz | tar xJ -C .deps ./usr/include
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DARMADILLO_INCLUDE_DIR="$PWD/.deps/usr/include" -DBUILD_TESTING=ON
srun -n1 -c4 --mem=6G cmake --build build -j4
srun -n1 -c1 --mem=2G ctest --test-dir build --output-on-failure
build/fsblup --version
tools/package.sh build dist
```

仓库按 GPL-3.0-or-later 分发。R 包保持独立，`.Rbuildignore` 排除 cpp。发布包包含对应 C++ 源码、夹具、复验脚本及第三方许可。
