# fsblup 0.3.0

Linux 命令行工具，用于单性状、每个个体一条表型记录的融合核 BLUP。将候选个体保留在表型文件里，性状填 `NA`。在 `predictions.tsv` 中按 `role=candidate` 取其带 ID 的 `u` 和 `yhat`。未列入表型文件的系谱祖先可以参与构核。

## 直接运行

发布包适用于 x86_64 Linux、glibc 2.34 及以上。解包后保留 `bin/` 和 `lib/` 的相对位置。程序、预测与调参均可在无 R 环境运行。0.3.0 在 CentOS Stream 9 构建，已在无 R 的 Red Hat UBI 9 Minimal 独立根文件系统（glibc 2.34）实测 nearPD 和多工作线程调参。以下命令在解包目录执行，输出目录使用新名字。源码树中可分别设置 `exe=build/fsblup` 和 `d=tests/data`。

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
- 调参先限制三层源矩阵到参考 ID，再融合。对所有已有参考群评分的合格权重按评分顺序检查全群融合，取第一个能构造有效协方差的权重，再用全部有表型个体重拟合。全群可行性检查只读取核。`search.tsv` 写入各点的参考群评分和全群检查状态，`tuning.tsv` 同时给出参考群最优、最终权重及两者的评分。覆盖在参考集中退化时可指定固定权重。独立评估请将留出个体的拟合性状填为 NA，并在预测后单独评分。预构核可含所有样本的无监督特征统计量。
- `--threads N` 默认设置 OpenBLAS 内层线程数（`--parallel blas`），权重按序评估。自动调参可选 `--parallel weights --threads N`，一个工作线程完整计算一个权重的融合核和所有训练折。主线程按原顺序归并同一轮的结果；进程级 BLAS 线程数在启动时设为 1。Slurm 示例见下。
- `--memory-mb` 默认 8192 MiB，控制密集工作集估算。外层并行会额外预留共享的全群与参考群数据、条件块、设计矩阵，以及每个并发任务私有的 K、谱分解/求解空间、线程与 BLAS 缓冲，计入输出缓冲。预算不足会减少并发数；单个候选也放不下时明确报错。`worker_memory.tsv` 给出共享量、每任务量和选定并发数。`runtime.tsv` 的 threads 来自 OpenBLAS，weight_workers 为初始网格实际执行的工作者数，包含主线程；细化轮最多使用当轮去重后的候选数。进程实际峰值 RSS 另列，Slurm 仍约束整个作业。
- `fusion.tsv` 记录覆盖、权重和扰动。`fuse` 或 `fit --write-k` 输出完整 `K_raw.tsv`、`K.tsv` 和 ID。模型文件含 Vu、Ve、lambda、边界状态；`fixed.tsv` 含各固定效应。搜索输出 `search.tsv` 与 `tuning.tsv`。若所有网格点失格，检查保留的搜索轨迹和 `failure.txt`。运行结果以目录为单位发布。
- 支持 A/G/O 同集合、A=G 且 O 为真子集，以及 O 严格包含于 G、G 严格包含于 A 的嵌套覆盖。其它情形会给出分组数量和示例 ID。输入约定、分块公式和实测边界见 [数值验收记录](VALIDATION.md)。

## Slurm 上的并行选权

下面的可重放例子在源码树 cpp 目录执行。使用独立发布包时，将 `build/fsblup` 改为 `bin/fsblup`，并从 `source/cpp/tests/nested` 读取夹具。

```bash
d=tests/nested
OPENBLAS_NUM_THREADS=1 srun -n1 -c4 --cpu-bind=cores --mem=2G build/fsblup fit \
  --A "$d/A.tsv" --G "$d/G.tsv" --O "$d/O.tsv" --pheno "$d/pheno.tsv" \
  --id ID --trait trait --folds-file "$d/folds.tsv" --metric rmse --strict-simplex \
  --pd-policy nearPD --parallel weights --threads 4 --out nested-parallel
```

本机 192 个体的嵌套计时用例从 1 个权重工作者增加到 8 个，端到端中位数由约 0.52 s 降至 0.19 s，峰值 RSS 同时增加。固定权重的大核投影可用 `--threads` 直接评估内层 BLAS 并行。原始数据构核的读写时间占比较大。[逐配置性能、调试结果与复现命令](PERFORMANCE.md)

## 融合矩阵的正定性

`fuse` 和 `fit` 接受 `--pd-policy strict|nearPD`。默认 `strict` 保留既有行为。它检查原矩阵，接受正定、合法半正定及舍入级负特征值，对实质不定矩阵报错。启用 `nearPD` 后，每个待处理块先加一次 `1e-6 I` 并尝试 Cholesky，成功即使用该块。失败时运行原生 C++ 的 `Matrix::nearPD` 协方差算法，验证收敛与 Cholesky 后交给后续计算。两种策略均适用于最终 K 及 B 模式实际参与传播的 Gss。C 模式用作回归条件的原始系谱子块仍须可做 Cholesky。

默认投影允许方差随协方差一起变化。令 S 为该块加过扰动的矩阵，P 为交给下一步的矩阵。`pd.tsv` 的每行记录块名、策略、动作、扰动、迭代与收敛状态、原始和使用矩阵的最小特征值、`normF=‖P−S‖F`、`relative_normF=normF/‖S‖F`、最大单元素变化及末次迭代相对变化。只加扰动时投影量为零，未计算的谱值和未执行的迭代状态写 NA。`K_raw.tsv` 位于最终 K 的正定处理之前；若 B 的中间 Gss 已投影，该影响已经进入 K_raw。`K.tsv` 是实际用于拟合的全群核。搜索在参考群按每组权重构核，`search.tsv` 的 nearpd_blocks、max_relative_normF 属于参考群。full_status 记录各点的全群检查状态，选中点另有 full_nearpd_blocks、full_max_relative_normF；逐块明细见全群 pd.tsv。

官方 example.zip 的完整嵌套示例可按以下方式运行。age 的 198 个表型个体都有基因型，参考群自动选权会触发覆盖检查。固定权重可以使用 1,407 个个体构核并为表型文件中的个体输出预测。

```bash
d=example-input
build/fsblup fit --A "$d/test_matA.csv" --G "$d/test_matG.csv" --O "$d/test_matO.csv" \
  --pheno "$d/test_pheno.csv" --id id --trait age --alpha .2 --beta .5 \
  --pd-policy nearPD --threads 1 --write-k --out example-nearPD
```

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
